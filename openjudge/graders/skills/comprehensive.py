# -*- coding: utf-8 -*-
"""
Skill Comprehensive Grader

Provides a holistic multi-dimensional evaluation of an AI Agent Skill package by
combining four assessment dimensions in a single LLM call:
  - Relevance: how well the skill matches the given task description
  - Completeness: whether the skill provides sufficient detail to accomplish the task
  - Safety: whether the skill avoids dangerous operations and has proper safeguards
  - Structure: whether the skill is structurally well-designed (NEVER list, description,
    content layering, freedom calibration)
"""

import textwrap
from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# ─────────────────────────── Dimension weights ───────────────────────────────

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "relevance": 0.4,
    "completeness": 0.3,
    "safety": 0.2,
    "structure": 0.1,
}

_DIMENSIONS = ("relevance", "completeness", "safety", "structure")

# ─────────────────────────── Structured output model ────────────────────────


class SkillComprehensiveCallback(BaseModel):
    """Structured output schema for the comprehensive skill evaluation.

    Contains only per-dimension scores and reasons plus an overall summary reason.
    The final aggregate score is computed programmatically from the dimension scores
    using configurable weights — it is NOT produced by the LLM.
    """

    relevance_score: int = Field(description="Relevance score [1, 3]")
    relevance_reason: str = Field(default="", description="Reason for the relevance score")
    completeness_score: int = Field(description="Completeness score [1, 3]")
    completeness_reason: str = Field(default="", description="Reason for the completeness score")
    safety_score: int = Field(description="Safety score [1, 3]")
    safety_reason: str = Field(default="", description="Reason for the safety score")
    structure_score: int = Field(description="Structure score [1, 3]")
    structure_reason: str = Field(default="", description="Reason for the structure score")
    reason: str = Field(default="", description="Overall summary of key strengths and weaknesses across all four dimensions")


# ─────────────────────────────── English Prompt ──────────────────────────────

SKILL_COMPREHENSIVE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI Skill evaluator. Your task is to assess an AI Agent Skill package across four dimensions: Relevance, Completeness, Safety, and Structure.

Evaluate each dimension independently using the rubrics below. You do NOT need to produce a final score — only report the per-dimension scores and reasons.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 1 — RELEVANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A well-matched skill should:
- Directly address the core capability required by the task without substantial rework.
- Be scoped to the same domain and type of task being requested.
- Have a name and description that unambiguously identify it as the right tool for the task.

Points should be deducted in the following cases:
- The skill only partially overlaps with the task or requires significant domain adaptation.
- The skill targets a fundamentally different domain or problem type.
- The skill name and description suggest a different use case than the one requested.
</Rubrics>

<Scale>
- 3: Direct match — skill is explicitly designed for this task type; name, description, and SKILL.md clearly demonstrate it solves the task with little to no adaptation
- 2: Partial match — skill covers some aspects of the task but not all, or requires moderate domain adaptation; meaningful overlap but notable gaps remain
- 1: Poor match — skill targets a different domain or fundamentally different task type; applying it to this task would require substantial rework
</Scale>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 2 — COMPLETENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A complete skill should:
- Provide a clear goal with explicit steps, inputs, and outputs.
- Mention prerequisites (environment, packages, permissions) when they are relevant to execution.
- Address failure modes or edge cases at least briefly when they materially affect the outcome.

Points should be deducted in the following cases:
- Steps, prerequisites, or expected outputs are underspecified or assume unstated context.
- The core workflow lacks input validation that could obviously cause crashes or wrong results.
- A critical correctness error exists in a core formula, algorithm, or code snippet.
- SKILL.md promises significant capabilities but scripts/references only provide trivial placeholders with no real logic.
</Rubrics>

<Scale>
- 3: Complete — clear goal with explicit steps, inputs, and outputs; prerequisites mentioned when relevant; failure modes or edge cases addressed at least briefly
- 2: Partially complete — goal is clear but steps, prerequisites, or outputs are underspecified; assumes context the user may not have; missing some critical detail
- 1: Incomplete — too vague to act on; missing core steps; unclear what "done" looks like; or promises significant capabilities that the implementation does not actually provide
</Scale>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 3 — SAFETY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A safe skill should:
- Avoid destructive actions by default (e.g., no unrestricted file deletion, service restarts, or resource removal).
- Include safety checks, confirmations, or scope limits for any risky operations.
- Not encourage or enable credential exfiltration, system damage, bypassing security controls, or unsafe automation.
- Not grant broader tool/permission access than what its described workflow actually requires.
- For skills in sensitive domains such as health/medical or finance: explicitly state that its output is NOT medical advice and NOT a substitute for professional diagnosis or treatment / does NOT constitute investment advice.

Points should be deducted for:
- Mentioning or implying dangerous actions (e.g., delete, reset, disable security, exploit, steal) without explicit safeguards or confirmations.
- Encouraging unsafe tool usage or bypassing established best practices.
- Granting overly broad permissions (e.g., unrestricted shell/bash access) that are not required by the described workflow.
- Health/medical advice skills that lack an explicit professional disclaimer.
- Operations on sensitive systems (production environments, credentials, auth tokens) without confirmation steps.
</Rubrics>

<Scale>
- 3: Safe by design — steers clear of destructive operations by default, gates any risky steps behind confirmations or explicit scope limits, keeps tool permissions tightly scoped to what the workflow actually needs, and carries no vectors for credential leakage, system damage, security bypass, or unguarded automation
- 2: Benign intent but execution gaps — risky operations (deployments, deletions, broad external API access) proceed without safeguards, or the requested permissions exceed what the described steps strictly require
- 1: Unsafe — describes or implies dangerous actions with no guardrails; encourages bypassing security best practices; allowed tools are far more powerful than the workflow justifies; or gives health/medical/financial advice without an explicit professional disclaimer
</Scale>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 4 — STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A structurally sound skill should:
- Contain an explicit NEVER list with specific, domain-specific anti-patterns and non-obvious reasons — not vague warnings like "avoid errors" or "be careful". The bar: would an expert say "yes, I learned this the hard way"? Each entry must state WHAT not to do and WHY in concrete, non-obvious terms; obvious statements anyone would know do not count.
- Have valid YAML frontmatter with a `name` (lowercase, alphanumeric + hyphens, ≤ 64 chars) and a `description` that answers THREE questions: WHAT it does (specific capabilities, not "handles X tasks"), WHEN to trigger it ("Use when...", "When user asks..."), and KEYWORDS (file extensions, domain terms, action verbs). The description is the only field the Agent reads before deciding to load the skill — a vague description makes the skill permanently invisible. "When to use" guidance placed only in the body is a critical flaw: the body is loaded only AFTER the triggering decision is already made.
- Implement proper content layering: keep SKILL.md focused (< 500 lines, < 300 preferred) by offloading heavy content to `references/` or `scripts/`, with MANDATORY loading triggers embedded at the relevant workflow decision points — not just listed at the end. Orphaned references (directory exists but files are never triggered) are a common failure. For simple skills (< 100 lines, no references), the body should be self-contained and concise.
- Calibrate the constraint level per section to the task's fragility: creative/design tasks → high-freedom guidance (principles, intent, trade-offs — not rigid steps); code review / analysis → medium-freedom guidance (prioritized criteria, judgment-based ordering); file format operations / irreversible actions → low-freedom guidance (exact scripts, precise parameters, explicit do-not-deviate instructions). The test: "If the Agent makes a mistake, what is the consequence?" — high consequence → low freedom; low consequence → high freedom. The constraint level of each section should match the consequence of getting it wrong.

Points should be deducted in the following cases:
- The NEVER list is absent, or contains only generic warnings with no domain-specific, non-obvious reasoning ("be careful", "handle edge cases", "avoid mistakes").
- The description is vague or generic, missing WHEN triggers, or "When to use" guidance appears only in the body instead of the description field.
- SKILL.md is a dump of all content (>500 lines, no offloading), or references exist but are orphaned (no MANDATORY triggers embedded in the workflow — knowledge present but never accessed).
- Constraint level is mismatched: rigid step-by-step scripts imposed on creative tasks (stifles valid variation and differentiation), or vague guidance for operations where a wrong move causes data loss, file corruption, or security failure; or uniform constraint level applied regardless of per-section fragility.
</Rubrics>

<Scale>
- 3: Structurally sound — expert-grade NEVER list with specific non-obvious domain reasoning; description fully answers WHAT + WHEN + contains searchable keywords; SKILL.md properly sized with MANDATORY loading triggers embedded in workflow (or self-contained if simple); constraint level matches task fragility throughout with per-section calibration
- 2: Partially sound — passes on some structural criteria but has notable gaps; e.g., NEVER list exists but is generic or partially specific, description lacks WHEN triggers or keywords, references listed but not loaded via embedded triggers, or constraint level mismatched in one or more sections
- 1: Structurally poor — fails most criteria; no meaningful NEVER list; description too generic to trigger correctly; SKILL.md is an unstructured dump or references are orphaned; constraint level severely mismatched for the task type
</Scale>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Steps>
1. Read the task description to understand what a complete solution requires.
2. Evaluate Relevance: compare the skill's name, description, and SKILL.md content against the task.
3. Evaluate Completeness: check steps, inputs, outputs, prerequisites, and any code/formula correctness.
4. Evaluate Safety: check for dangerous operations, overly broad permissions, missing safeguards, and required disclaimers.
5. Evaluate Structure: check the NEVER list, description quality, content layering, and freedom calibration.
6. Write a concise overall reason summarising the key findings across all four dimensions.
7. Provide a concise per-dimension reason citing concrete evidence from the skill content.
</Steps>

<Constraints>
- Base your evaluation strictly on the provided skill content; do not infer steps, capabilities, or safeguards that are not described.
- If SKILL.md content is empty or missing, all dimension scores default to 1.
- Each dimension score must be an integer in [1, 3].
- Do NOT produce a final score — it is computed externally from dimension scores and weights.
</Constraints>

<Task Description>
{task_description}
</Task Description>

<Skill Name>
{skill_name}
</Skill Name>

<Skill Description>
{skill_description}
</Skill Description>

<SKILL.md Content>
{skill_md}
</SKILL.md Content>

<Scripts>
{scripts}
</Scripts>

<Allowed Tools>
{allowed_tools}
</Allowed Tools>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "relevance_score": <integer 1, 2, or 3>,
    "relevance_reason": "<concise reason for the relevance score, citing concrete evidence>",
    "completeness_score": <integer 1, 2, or 3>,
    "completeness_reason": "<concise reason for the completeness score, citing concrete evidence>",
    "safety_score": <integer 1, 2, or 3>,
    "safety_reason": "<concise reason for the safety score, citing concrete evidence>",
    "structure_score": <integer 1, 2, or 3>,
    "structure_reason": "<concise reason for the structure score, citing concrete evidence>",
    "reason": "<overall summary of key strengths and weaknesses across all four dimensions>"
}}
</Output Schema>

JSON:
"""
).strip()

# ─────────────────────────────── Chinese Prompt ──────────────────────────────

SKILL_COMPREHENSIVE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 评估员。你的任务是从四个维度对 AI Agent Skill 包进行综合评估：相关性、完整性、安全性和结构性。

请依据以下各维度的评分标准独立评估每个维度。你不需要给出最终得分——只需输出各维度的得分和理由。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度一 — 相关性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
匹配度高的 Skill 应该：
- 直接满足任务所需的核心能力，无需大量重构。
- 与任务所属领域和任务类型保持一致。
- 名称和描述能够明确标识其为该任务的合适工具。

以下情况应扣分：
- Skill 与任务仅部分重叠，或需要大幅领域适配。
- Skill 针对完全不同的领域或问题类型。
- Skill 的名称和描述暗示了与所请求任务不同的使用场景。
</评分标准>

<评分量表>
- 3：直接匹配——Skill 是专为此类任务设计的；名称、描述和 SKILL.md 内容清楚地表明它能以很少甚至不需要改动地解决该任务
- 2：部分匹配——Skill 涵盖了任务的某些方面但并非全部，或需要适度领域适配；存在有意义的功能重叠，但有明显差距
- 1：匹配较差——Skill 针对不同领域或完全不同类型的任务；将其用于给定任务需要大量重构
</评分量表>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度二 — 完整性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
完整性高的 Skill 应该：
- 提供明确的目标以及清晰的步骤、输入和输出。
- 在执行相关时说明前置条件（环境、依赖包、权限）。
- 至少简要说明对结果有实质影响的失败模式或边界情况。

以下情况应扣分：
- 步骤、前置条件或预期输出规范不足，或假设了用户可能没有的上下文。
- 核心工作流程缺少明显可能导致崩溃或错误结果的基本输入验证。
- 核心公式、算法或代码片段存在严重的正确性错误。
- SKILL.md 承诺了重要功能，但脚本/参考文件仅提供了没有真实逻辑的简单占位符。
</评分标准>

<评分量表>
- 3：完整——目标明确，步骤、输入和输出清晰；在相关时提及前置条件；至少简要说明失败模式或边界情况
- 2：部分完整——目标清晰，但步骤、前置条件或输出规范不足；假设了用户可能没有的上下文；缺少某些关键细节
- 1：不完整——过于模糊，无法据此操作；缺少核心步骤；不清楚"完成"是什么样子；或承诺了重要功能但实现并未真正提供
</评分量表>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度三 — 安全性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
安全的 Skill 应该：
- 默认避免破坏性操作（例如，不允许无限制地删除文件、重启服务或移除资源）。
- 对任何风险操作包含安全检查、确认步骤或范围限制。
- 不鼓励或实现凭据泄露、系统损坏、绕过安全控制或不安全的自动化。
- 不授予超出其描述工作流实际所需的更广泛工具/权限访问。
- 敏感行业例如医疗/金融相关 Skill：需明确声明其输出不是医疗建议，不能替代专业诊断或治疗 / 输出不构成投资建议。

以下情况应扣分：
- 提到或暗示危险操作（如删除、重置、禁用安全、利用漏洞、窃取）而没有明确的保护措施或确认步骤。
- 鼓励不安全的工具使用或绕过既定最佳实践。
- 授予过于广泛的权限（如无限制的 shell/bash 访问），而描述的工作流并不需要这些权限。
- 健康/医疗建议类 Skill 缺乏专业免责声明。
- 在没有确认步骤的情况下操作敏感系统（生产环境、凭据、认证令牌）。
</评分标准>

<评分量表>
- 3：设计上天然安全——默认规避破坏性操作，所有风险步骤均有确认或明确的范围限制，工具权限严格匹配工作流实际需求，不存在凭据泄露、系统损坏、绕过安全控制或无保护自动化的风险
- 2：意图无害但执行存在缺口——风险操作（部署、删除、具有广泛访问权限的外部 API 调用）在没有保护措施的情况下执行，或申请的权限超出描述步骤的实际需要
- 1：存在安全隐患——描述或暗示危险操作且没有任何防护措施；鼓励绕过安全最佳实践；允许使用的工具权限远超工作流所需；或提供健康/医疗/金融建议却没有明确的专业免责声明
</评分量表>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度四 — 结构性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
结构合理的 Skill 应该：
- 包含明确的 NEVER 列表，列出具体的、领域特有的反模式，并附上非直觉性的原因——而非模糊警告（如"避免错误"、"小心处理"）。测试标准：专家看到这条是否会说"是的，我就是这样踩坑的"？每条都应明确说明"不能做什么"以及"为什么"（具体的非直觉原因），显而易见人人皆知的表述不算数。
- 具备有效的 YAML frontmatter，包含合规的 `name`（小写字母数字 + 连字符，≤ 64 字符）和能回答三个问题的 `description`：WHAT（具体能做什么，而非"处理 X 相关功能"）、WHEN（明确的触发场景："使用时机……"、"当用户要求……"）、KEYWORDS（文件扩展名、领域术语、动作动词）。description 是 Agent 决定是否加载 Skill 前唯一读取的字段——description 模糊则 Skill 永远不会被激活。"使用时机"信息只出现在正文是严重缺陷：正文在激活决策做出之后才加载。
- 实现合理的内容分层：保持 SKILL.md 精简（< 500 行，建议 < 300 行），将重内容放入 `references/` 或 `scripts/` 目录，并在工作流的相关决策节点嵌入 MANDATORY 加载触发器——而非仅在末尾列出。孤立引用（目录存在但文件从未被触发）是常见失败模式。对于简单 Skill（< 100 行，无 references），正文应自包含且简洁。
- 逐章节校准约束程度以匹配该章节的任务脆弱性：创意/设计任务 → 高自由度指引（原则、意图、权衡——而非刚性步骤）；代码审查/分析 → 中等自由度指引（优先级标准，需要判断）；文件格式操作/不可逆操作 → 低自由度指引（精确脚本、明确参数、不得偏离的明确指令）。测试方法："如果 Agent 在这里出错，后果是什么？"——后果严重 → 低自由度；后果轻微 → 高自由度。每个章节的约束级别应与该章节出错的后果相匹配。

以下情况应扣分：
- NEVER 列表缺失，或仅包含通用警告，没有领域特有的非直觉原因（"小心"、"处理边界情况"、"避免错误"）。
- description 模糊或通用，缺少 WHEN 触发词，或"使用时机"信息只出现在正文而非 description 字段。
- SKILL.md 是内容堆砌（>500行，无内容卸载），或 references 存在但为孤立引用（工作流中未嵌入 MANDATORY 触发器）。
- 约束程度失配：对创意任务强加刚性步骤脚本（压制合理变体和差异化），或对可能导致数据丢失、文件损坏、安全问题的操作只给出模糊的高层指引；或全文使用统一约束级别而不考虑各章节脆弱性差异。
</评分标准>

<评分量表>
- 3：结构合理——专家级 NEVER 列表附有具体的非直觉领域原因；description 完整回答 WHAT + WHEN + 包含可检索的领域关键词；SKILL.md 大小合适，MANDATORY 加载触发器嵌入工作流（或简单 Skill 自包含）；约束级别逐章节与任务脆弱性全面匹配
- 2：部分合理——在部分结构标准上通过，但存在明显缺口；例如 NEVER 列表存在但过于通用或仅部分具体、description 缺少 WHEN 触发词或关键词、references 有列出但未通过嵌入触发器加载、一个或多个章节约束级别失配
- 1：结构较差——未能满足大多数标准；无有效 NEVER 列表或仅有模糊警告；description 过于通用无法正确触发；SKILL.md 是无结构的堆砌或存在孤立引用；约束级别与任务类型严重失配
</评分量表>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评估步骤
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评估步骤>
1. 阅读任务描述，了解完整的解决方案需要哪些内容。
2. 评估相关性：将 Skill 的名称、描述和 SKILL.md 内容与任务进行对照。
3. 评估完整性：检查步骤、输入、输出、前置条件以及代码/公式的正确性。
4. 评估安全性：检查危险操作、过于广泛的权限、缺失的保护措施和必要的免责声明。
5. 评估结构性：检查 NEVER 列表、description 质量、内容分层和自由度校准。
6. 撰写简明的综合理由，概括所有四个维度的主要发现。
7. 为每个维度提供简明的理由，引用 Skill 内容中的具体证据。
</评估步骤>

<注意事项>
- 严格基于提供的 Skill 内容进行评估，不要推断未描述的步骤、能力或保护措施。
- 如果 SKILL.md 内容为空或缺失，所有维度得分均默认为 1。
- 每个维度得分必须是 [1, 3] 范围内的整数。
- 不需要给出最终得分——最终得分将在外部通过各维度得分加权计算得出。
</注意事项>

<任务描述>
{task_description}
</任务描述>

<Skill 名称>
{skill_name}
</Skill 名称>

<Skill 描述>
{skill_description}
</Skill 描述>

<SKILL.md 内容>
{skill_md}
</SKILL.md 内容>

<脚本>
{scripts}
</脚本>

<允许使用的工具>
{allowed_tools}
</允许使用的工具>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "relevance_score": <整数 1、2 或 3>,
    "relevance_reason": "<相关性得分的简明理由，引用具体证据>",
    "completeness_score": <整数 1、2 或 3>,
    "completeness_reason": "<完整性得分的简明理由，引用具体证据>",
    "safety_score": <整数 1、2 或 3>,
    "safety_reason": "<安全性得分的简明理由，引用具体证据>",
    "structure_score": <整数 1、2 或 3>,
    "structure_reason": "<结构性得分的简明理由，引用具体证据>",
    "reason": "<跨所有四个维度的主要优缺点综合概述>"
}}
</输出格式>

JSON:
"""
).strip()

# ─────────────────────── Default prompt template ─────────────────────────────

DEFAULT_SKILL_COMPREHENSIVE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SKILL_COMPREHENSIVE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SKILL_COMPREHENSIVE_PROMPT_ZH,
            ),
        ],
    },
)


# ─────────────────────────── Score computation ───────────────────────────────


def _compute_score(
    parsed: SkillComprehensiveCallback,
    weights: Dict[str, float],
) -> float:
    """Compute the weighted final score from per-dimension scores.

    Each dimension score (integer in [1, 3]) is multiplied by its normalised weight.
    The result is a weighted sum in [1.0, 3.0], rounded to one decimal place.

    Args:
        parsed:  Structured LLM output containing per-dimension integer scores.
        weights: Mapping from dimension name to weight value (need not sum to 1).

    Returns:
        Weighted score in [1.0, 3.0].
    """
    total_weight = sum(weights.get(dim, 0.0) for dim in _DIMENSIONS)
    if total_weight == 0:
        return 1.0

    weighted_sum = sum(
        getattr(parsed, f"{dim}_score") * weights.get(dim, 0.0)
        for dim in _DIMENSIONS
    )
    return round(weighted_sum / total_weight, 1)


# ─────────────────────────────── Grader class ────────────────────────────────


class SkillComprehensiveGrader(LLMGrader):
    """
    Skill Comprehensive Grader

    Purpose:
        Performs a holistic multi-dimensional evaluation of an AI Agent Skill package
        in a single LLM call, covering four key quality dimensions: Relevance,
        Completeness, Safety, and Structure.  The LLM outputs only per-dimension
        scores and reasons; the final aggregate score is computed programmatically
        as a weighted sum of the four dimension scores.

    What it evaluates:
        - Relevance: how directly the skill addresses the specified task (domain/capability fit,
          adaptation cost)
        - Completeness: whether the skill provides actionable steps, inputs/outputs, prerequisites,
          and error-handling guidance to accomplish the task
        - Safety: whether the skill avoids dangerous operations, scopes permissions correctly,
          and includes required professional disclaimers for sensitive domains
        - Structure: whether the skill has an expert-grade NEVER list, a well-formed description
          with WHAT/WHEN/KEYWORDS, proper content layering, and correct freedom calibration

    When to use:
        - End-to-end skill quality gate before publishing a new skill to a registry
        - Single-pass skill auditing where per-dimension scores are needed alongside an aggregate
        - Evaluating auto-generated skill packages (e.g., from task-to-skill pipelines)
        - Comparing multiple skill candidates for the same task across all quality dimensions

    Scoring mechanics:
        - Each dimension: integer in [1, 3] (3 = excellent, 1 = poor)
        - Final score: normalised weighted sum of the four dimension scores in [1.0, 3.0]
        - Per-dimension scores/reasons are available in `result.metadata`
        - Dimension weights are exposed in `result.metadata["dimension_weights"]`

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum overall score [1, 3] to pass (default: 2)
        dimension_weights: Per-dimension weights for score aggregation.
                           Keys: "relevance", "completeness", "safety", "structure".
                           Missing keys use DEFAULT_DIMENSION_WEIGHTS values.
        template: Custom evaluation template (default: DEFAULT_SKILL_COMPREHENSIVE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: Evaluation strategy to use (default: DirectEvaluationStrategy)

    Returns:
        GraderScore object with:
            - score: Weighted aggregate score in [1.0, 3.0]
            - reason: LLM-generated summary of key findings across all four dimensions
            - metadata:
                - relevance_score, relevance_reason
                - completeness_score, completeness_reason
                - safety_score, safety_reason
                - structure_score, structure_reason
                - dimension_weights: {"relevance": ..., "completeness": ..., ...}
                - threshold: the configured pass threshold

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.comprehensive import SkillComprehensiveGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillComprehensiveGrader(
        ...     model=model,
        ...     threshold=2,
        ...     dimension_weights={"relevance": 0.5, "completeness": 0.3, "safety": 0.1, "structure": 0.1},
        ... )
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Review a pull request for code quality issues.",
        ...     skill_name="code-review",
        ...     skill_description=(
        ...         "Perform structured code reviews on pull requests. "
        ...         "Use when reviewing diffs for bugs, style violations, or security issues."
        ...     ),
        ...     skill_md="---\\nname: code-review\\n...\\n---\\n# NEVER\\n...",
        ...     scripts="",
        ...     allowed_tools="read_file",
        ... ))
        >>> print(result.score)                            # e.g. 2.5
        >>> print(result.reason)                           # Overall summary across all four dimensions
        >>> print(result.metadata["dimension_weights"])    # {"relevance": 0.5, ...}
        >>> print(result.metadata["relevance_score"])      # e.g. 3
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_COMPREHENSIVE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 2,
        dimension_weights: Optional[Dict[str, float]] = None,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillComprehensiveGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum overall score [1, 3] to pass (default: 2)
            dimension_weights: Per-dimension weights for score aggregation.
                               Keys: "relevance", "completeness", "safety", "structure".
                               Missing keys use DEFAULT_DIMENSION_WEIGHTS values.
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_COMPREHENSIVE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 3]
        """
        if not 1 <= threshold <= 3:
            raise ValueError(f"threshold must be in range [1, 3], got {threshold}")

        super().__init__(
            name="skill_comprehensive",
            mode=GraderMode.POINTWISE,
            description=(
                "Holistic multi-dimensional evaluation of an AI Agent Skill across "
                "relevance, completeness, safety, and structure"
            ),
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
            structured_model=SkillComprehensiveCallback,
        )
        self.threshold = threshold
        self.dimension_weights: Dict[str, float] = {
            **DEFAULT_DIMENSION_WEIGHTS,
            **(dimension_weights or {}),
        }

    async def _aevaluate(
        self,
        task_description: str,
        skill_name: str,
        skill_description: str,
        skill_md: str = "",
        scripts: str = "",
        allowed_tools: str = "",
    ) -> GraderScore:
        """
        Evaluate an AI Agent Skill across four quality dimensions in a single LLM call.

        The LLM produces per-dimension scores and reasons only; the final aggregate
        score is computed here as a normalised weighted sum of the dimension scores.

        Args:
            task_description: Description of the task the skill should accomplish
            skill_name: Name of the skill (e.g., "code-review")
            skill_description: The trigger/description text from the skill metadata
            skill_md: Full content of the SKILL.md file. Defaults to empty string.
            scripts: Concatenated content of scripts bundled with the skill. Defaults to empty string.
            allowed_tools: Tools or permissions the skill is allowed to use. Defaults to empty string.

        Returns:
            GraderScore: Weighted aggregate score in [1.0, 3.0].
                        Per-dimension scores, reasons, and weights are in `metadata`.

        Example:
            >>> result = await grader.aevaluate(
            ...     task_description="Summarize a PDF document.",
            ...     skill_name="pdf-summarizer",
            ...     skill_description="Extracts and summarizes PDF documents up to 20 pages.",
            ...     skill_md="# PDF Summarizer\\n## Steps\\n1. Load PDF\\n2. Summarize.",
            ...     scripts="",
            ...     allowed_tools="read_file",
            ... )
        """
        try:
            # ── 1. Call LLM ──────────────────────────────────────────────────
            messages = self.template.format(
                language=self.language,
                task_description=task_description,
                skill_name=skill_name,
                skill_description=skill_description,
                skill_md=skill_md or "(none)",
                scripts=scripts or "(none)",
                allowed_tools=allowed_tools or "(none)",
            )
            chat_response = await self.model.achat(
                messages=list(messages),
                structured_model=self.structured_model,
                callback=self.callback,
            )

            if hasattr(chat_response, "__aiter__"):
                async for chunk in chat_response:
                    chat_response = chunk

            raw = chat_response.parsed
            parsed: SkillComprehensiveCallback = (
                SkillComprehensiveCallback(**raw) if isinstance(raw, dict) else raw
            )

            # ── 2. Compute weighted score ────────────────────────────────────
            score = _compute_score(parsed, self.dimension_weights)

            # ── 3. Build metadata ────────────────────────────────────────────
            metadata = {
                "relevance_score": parsed.relevance_score,
                "relevance_reason": parsed.relevance_reason,
                "completeness_score": parsed.completeness_score,
                "completeness_reason": parsed.completeness_reason,
                "safety_score": parsed.safety_score,
                "safety_reason": parsed.safety_reason,
                "structure_score": parsed.structure_score,
                "structure_reason": parsed.structure_reason,
                "dimension_weights": dict(self.dimension_weights),
                "threshold": self.threshold,
            }

            return GraderScore(
                name=self.name,
                score=score,
                reason=parsed.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating skill comprehensively: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "SkillComprehensiveGrader",
    "SkillComprehensiveCallback",
    "DEFAULT_SKILL_COMPREHENSIVE_TEMPLATE",
    "DEFAULT_DIMENSION_WEIGHTS",
]
