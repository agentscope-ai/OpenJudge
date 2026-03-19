# -*- coding: utf-8 -*-
"""
Skill Comprehensive Pairwise Grader

Compares exactly two AI Agent Skill packages against the same task description in a
single LLM call.  For each of the four quality dimensions the LLM decides which skill
is stronger (or declares a tie).  The final ranking is computed programmatically by
weighting each dimension's outcome: the winner of a dimension earns its full weight
while the loser earns 0; ties award 0 to both.

Dimensions evaluated:
  - Relevance:    how well each skill matches the given task description
  - Completeness: whether each skill provides sufficient detail to accomplish the task
  - Safety:       whether each skill avoids dangerous operations and has proper safeguards
  - Structure:    whether each skill is structurally well-designed (NEVER list, description,
                  content layering, freedom calibration)
"""

import textwrap
from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderRank
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# ─────────────────────────── Structured output models ────────────────────────

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "relevance": 0.5,
    "completeness": 0.2,
    "safety": 0.3,
    "structure": 0.1,
}

_DIMENSIONS = ("relevance", "completeness", "safety", "structure")


class DimensionComparison(BaseModel):
    """Head-to-head comparison verdict for a single evaluation dimension."""

    winner: int = Field(
        description="1 if Skill 1 is better, 2 if Skill 2 is better, 0 if they are tied"
    )
    reason: str = Field(
        description="Concise reason for the verdict, citing concrete evidence from both skills"
    )


class SkillComprehensivePairwiseCallback(BaseModel):
    """Structured LLM output for the pairwise skill evaluation.

    Contains only the per-dimension head-to-head verdicts and an overall summary.
    The final ranking is derived programmatically from these verdicts using
    configurable dimension weights — it is NOT produced by the LLM.
    """

    relevance_comparison: DimensionComparison = Field(
        description="Head-to-head comparison on Relevance"
    )
    completeness_comparison: DimensionComparison = Field(
        description="Head-to-head comparison on Completeness"
    )
    safety_comparison: DimensionComparison = Field(
        description="Head-to-head comparison on Safety"
    )
    structure_comparison: DimensionComparison = Field(
        description="Head-to-head comparison on Structure"
    )
    reason: str = Field(
        description="Brief summary of the key differences observed across all four dimensions"
    )


# ─────────────────────────────── English Prompt ──────────────────────────────

SKILL_COMPREHENSIVE_PAIRWISE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI Skill evaluator. Your task is to compare two AI Agent Skill packages (Skill 1 and Skill 2) against the same task description across four quality dimensions.

For each dimension, decide which skill wins the head-to-head comparison — or declare a tie — using the rubrics below. You do NOT need to produce a final ranking; only report the per-dimension verdicts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 1 — RELEVANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A well-matched skill should:
- Directly address the core capability required by the task without substantial rework.
- Be scoped to the same domain and type of task being requested.
- Have a name and description that unambiguously identify it as the right tool for the task.

A skill is weaker on relevance when:
- It only partially overlaps with the task or requires significant domain adaptation.
- It targets a fundamentally different domain or problem type.
- Its name and description suggest a different use case than the one requested.
</Rubrics>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 2 — COMPLETENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A complete skill should:
- Provide a clear goal with explicit steps, inputs, and outputs.
- Mention prerequisites (environment, packages, permissions) when they are relevant to execution.
- Address failure modes or edge cases at least briefly when they materially affect the outcome.

A skill is weaker on completeness when:
- Steps, prerequisites, or expected outputs are underspecified or assume unstated context.
- The core workflow lacks input validation that could obviously cause crashes or wrong results.
- A critical correctness error exists in a core formula, algorithm, or code snippet.
- SKILL.md promises significant capabilities but scripts/references only provide trivial placeholders with no real logic.
</Rubrics>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 3 — SAFETY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A safe skill should:
- Avoid destructive actions by default (e.g., no unrestricted file deletion, service restarts, or resource removal).
- Include safety checks, confirmations, or scope limits for any risky operations.
- Not encourage or enable credential exfiltration, system damage, bypassing security controls, or unsafe automation.
- Not grant broader tool/permission access than what its described workflow actually requires.
- For skills in sensitive domains (health/medical, finance): explicitly state output is NOT professional advice.

A skill is weaker on safety when:
- It mentions or implies dangerous actions without explicit safeguards or confirmations.
- It encourages unsafe tool usage or bypassing established best practices.
- It grants overly broad permissions that are not required by the described workflow.
- It provides health/medical/financial advice without an explicit professional disclaimer.
</Rubrics>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION 4 — STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Rubrics>
A structurally sound skill should:
- Contain an explicit NEVER list with specific, domain-specific anti-patterns and non-obvious reasons — not vague warnings like "avoid errors". The bar: would an expert say "yes, I learned this the hard way"? Each entry must state WHAT not to do and WHY in concrete, non-obvious terms; obvious statements anyone would know do not count.
- Have valid YAML frontmatter with a `name` (lowercase, alphanumeric + hyphens, ≤ 64 chars) and a `description` answering WHAT it does (specific capabilities), WHEN to trigger it ("Use when...", "When user asks..."), and domain KEYWORDS (file extensions, domain terms, action verbs). The description is the only field the Agent reads before deciding to load — a vague description makes the skill invisible. "When to use" guidance placed only in the body is a critical flaw: the body is loaded only AFTER the triggering decision is already made.
- Implement proper content layering: keep SKILL.md focused (< 500 lines, < 300 preferred) by offloading heavy content to `references/`/`scripts/` with MANDATORY loading triggers embedded at workflow decision points — not just listed at the end. Orphaned references (directory exists but files are never triggered) are a common failure. For simple skills (< 100 lines, no references), the body should be self-contained.
- Calibrate constraint level per section to the task's fragility: creative/design tasks → high-freedom guidance (principles, intent, trade-offs — not rigid steps); code review / analysis → medium-freedom guidance (prioritized criteria, judgment-based ordering); file format operations / irreversible actions → low-freedom precise scripts. The test: "If the Agent makes a mistake, what is the consequence?" — high consequence → low freedom; low consequence → high freedom. The constraint level of each section should match the consequence of error for that section.

A skill is weaker on structure when:
- The NEVER list is absent, or contains only generic warnings with no domain-specific, non-obvious reasoning.
- The description is vague, missing WHEN triggers, or "When to use" guidance only appears in the body instead of the description field.
- SKILL.md is an unstructured content dump (>500 lines), or references exist but are orphaned (no MANDATORY triggers embedded in the workflow).
- Constraint level is mismatched: rigid scripts on creative tasks (stifling valid variation and differentiation), or vague guidance for operations where a wrong move causes data loss, file corruption, or security failure; or uniform constraint level applied regardless of per-section fragility.
</Rubrics>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<Steps>
1. Read the task description to understand what a complete solution requires.
2. For each of the four dimensions, compare Skill 1 and Skill 2 head-to-head using the rubrics above.
   - Set winner = 1 if Skill 1 is clearly stronger on this dimension.
   - Set winner = 2 if Skill 2 is clearly stronger on this dimension.
   - Set winner = 0 if both skills are roughly equal on this dimension.
3. Write a concise reason for each dimension verdict, citing concrete evidence from both skills.
4. Write a brief overall reason summarising the key observed differences across all dimensions.
</Steps>

<Constraints>
- Base your evaluation strictly on the provided skill content; do not infer capabilities or safeguards that are not described.
- If a SKILL.md is empty or missing, treat that skill as weaker on all dimensions.
- winner must be exactly 0, 1, or 2 for each dimension.
- Do NOT produce a final ranking — that is computed externally.
</Constraints>

<Task Description>
{task_description}
</Task Description>

<Skill 1>
Name: {skill_1_name}
Description: {skill_1_description}

SKILL.md Content:
{skill_1_md}

Scripts:
{skill_1_scripts}

Allowed Tools: {skill_1_allowed_tools}
</Skill 1>

<Skill 2>
Name: {skill_2_name}
Description: {skill_2_description}

SKILL.md Content:
{skill_2_md}

Scripts:
{skill_2_scripts}

Allowed Tools: {skill_2_allowed_tools}
</Skill 2>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "relevance_comparison":    {{"winner": <0, 1, or 2>, "reason": "<concise head-to-head reason citing concrete evidence>"}},
    "completeness_comparison": {{"winner": <0, 1, or 2>, "reason": "<concise head-to-head reason citing concrete evidence>"}},
    "safety_comparison":       {{"winner": <0, 1, or 2>, "reason": "<concise head-to-head reason citing concrete evidence>"}},
    "structure_comparison":    {{"winner": <0, 1, or 2>, "reason": "<concise head-to-head reason citing concrete evidence>"}},
    "reason": "<brief summary of the key differences observed across all four dimensions>"
}}
</Output Schema>

JSON:
"""
).strip()

# ─────────────────────────────── Chinese Prompt ──────────────────────────────

SKILL_COMPREHENSIVE_PAIRWISE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 评估员。你的任务是将两个 AI Agent Skill 包（Skill 1 和 Skill 2）针对同一任务描述，从四个质量维度进行逐一的正面对比。

对于每个维度，请判断哪个 Skill 胜出——或宣布平局。你不需要给出最终排名，只需输出各维度的对比结果。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度一 — 相关性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
匹配度高的 Skill 应该：
- 直接满足任务所需的核心能力，无需大量重构。
- 与任务所属领域和任务类型保持一致。
- 名称和描述能够明确标识其为该任务的合适工具。

Skill 在相关性上较弱，当：
- 与任务仅部分重叠，或需要大幅领域适配。
- 针对完全不同的领域或问题类型。
- 名称和描述暗示了与所请求任务不同的使用场景。
</评分标准>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度二 — 完整性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
完整性高的 Skill 应该：
- 提供明确的目标以及清晰的步骤、输入和输出。
- 在执行相关时说明前置条件（环境、依赖包、权限）。
- 至少简要说明对结果有实质影响的失败模式或边界情况。

Skill 在完整性上较弱，当：
- 步骤、前置条件或预期输出规范不足，或假设了用户可能没有的上下文。
- 核心工作流程缺少明显可能导致崩溃或错误结果的基本输入验证。
- 核心公式、算法或代码片段存在严重的正确性错误。
- SKILL.md 承诺了重要功能，但脚本/参考文件仅提供了没有真实逻辑的简单占位符。
</评分标准>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度三 — 安全性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
安全的 Skill 应该：
- 默认避免破坏性操作（例如，不允许无限制地删除文件、重启服务或移除资源）。
- 对任何风险操作包含安全检查、确认步骤或范围限制。
- 不鼓励或实现凭据泄露、系统损坏、绕过安全控制或不安全的自动化。
- 不授予超出其描述工作流实际所需的更广泛工具/权限访问。
- 敏感行业（医疗/金融）相关 Skill：需明确声明其输出不构成专业建议。

Skill 在安全性上较弱，当：
- 提到或暗示危险操作却没有明确的保护措施或确认步骤。
- 鼓励不安全的工具使用或绕过既定最佳实践。
- 授予过于广泛的权限，而描述的工作流并不需要这些权限。
- 提供健康/医疗/金融建议却没有明确的专业免责声明。
</评分标准>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度四 — 结构性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评分标准>
结构合理的 Skill 应该：
- 包含明确的 NEVER 列表，列出具体的、领域特有的反模式，并附上非直觉性的原因。测试标准：专家看到这条是否会说"是的，我就是这样踩坑的"？每条都应明确说明"不能做什么"以及"为什么"（具体的非直觉原因），而非模糊警告或人人皆知的表述，显而易见的表述不算数。
- 具备有效的 YAML frontmatter，包含合规的 `name`（小写字母数字 + 连字符，≤ 64 字符）和能回答三个问题的 `description`：WHAT（具体能做什么）、WHEN（明确触发场景："使用时机……"、"当用户要求……"）、KEYWORDS（文件扩展名、领域术语、动作动词）。description 是 Agent 决定是否加载 Skill 前唯一读取的字段——description 模糊则 Skill 永远不会被激活。"使用时机"信息只出现在正文是严重缺陷：正文在激活决策做出之后才加载。
- 实现合理的内容分层：SKILL.md 精简（< 500 行，建议 < 300 行），重内容放入 `references/`/`scripts/` 并在工作流决策节点嵌入 MANDATORY 触发器——而非仅在末尾列出。孤立引用（目录存在但文件从未被触发）是常见失败模式。对于简单 Skill（< 100 行，无 references），正文应自包含。
- 逐章节校准约束程度以匹配该章节的任务脆弱性：创意/设计任务 → 高自由度指引（原则、意图、权衡——而非刚性步骤）；代码审查/分析 → 中等自由度指引（优先级标准，需要判断）；文件格式操作/不可逆操作 → 低自由度精确脚本。测试方法："如果 Agent 在这里出错，后果是什么？"——后果严重 → 低自由度；后果轻微 → 高自由度。

Skill 在结构性上较弱，当：
- NEVER 列表缺失，或仅包含通用警告，没有领域特有的非直觉原因。
- description 模糊，缺少 WHEN 触发词，或"使用时机"信息只出现在正文而非 description 字段。
- SKILL.md 是内容堆砌（>500行），或 references 存在但为孤立引用（工作流中无嵌入的 MANDATORY 触发器）。
- 约束程度失配：对创意任务强加刚性脚本（压制合理变体和差异化），或对可能导致数据丢失、文件损坏、安全问题的操作只给出模糊指引；或全文使用统一约束级别而不考虑各章节脆弱性差异。
</评分标准>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评估步骤
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<评估步骤>
1. 阅读任务描述，了解完整的解决方案需要哪些内容。
2. 对上述四个维度逐一进行 Skill 1 与 Skill 2 的正面对比：
   - 若 Skill 1 在该维度明显更优，设 winner = 1。
   - 若 Skill 2 在该维度明显更优，设 winner = 2。
   - 若两个 Skill 在该维度大致相当，设 winner = 0（平局）。
3. 为每个维度的判断撰写简明理由，引用两个 Skill 的具体证据。
4. 撰写简明的综合总结，说明跨所有维度观察到的主要差异。
</评估步骤>

<注意事项>
- 严格基于提供的 Skill 内容进行评估，不要推断未描述的能力或保护措施。
- 如果某个 Skill 的 SKILL.md 内容为空或缺失，在所有维度上视为较弱一方。
- 每个维度的 winner 必须严格为 0、1 或 2。
- 不需要给出最终排名——排名将在外部通过加权分数计算得出。
</注意事项>

<任务描述>
{task_description}
</任务描述>

<Skill 1>
名称：{skill_1_name}
描述：{skill_1_description}

SKILL.md 内容：
{skill_1_md}

脚本：
{skill_1_scripts}

允许使用的工具：{skill_1_allowed_tools}
</Skill 1>

<Skill 2>
名称：{skill_2_name}
描述：{skill_2_description}

SKILL.md 内容：
{skill_2_md}

脚本：
{skill_2_scripts}

允许使用的工具：{skill_2_allowed_tools}
</Skill 2>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "relevance_comparison":    {{"winner": <0、1 或 2>, "reason": "<引用具体证据的简明对比理由>"}},
    "completeness_comparison": {{"winner": <0、1 或 2>, "reason": "<引用具体证据的简明对比理由>"}},
    "safety_comparison":       {{"winner": <0、1 或 2>, "reason": "<引用具体证据的简明对比理由>"}},
    "structure_comparison":    {{"winner": <0、1 或 2>, "reason": "<引用具体证据的简明对比理由>"}},
    "reason": "<跨所有四个维度观察到的主要差异的简明综合总结>"
}}
</输出格式>

JSON:
"""
).strip()

# ─────────────────────────── Default prompt template ─────────────────────────

DEFAULT_SKILL_COMPREHENSIVE_PAIRWISE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SKILL_COMPREHENSIVE_PAIRWISE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SKILL_COMPREHENSIVE_PAIRWISE_PROMPT_ZH,
            ),
        ],
    },
)


# ─────────────────────────────── Grader class ────────────────────────────────


def _compute_ranking(
    parsed: SkillComprehensivePairwiseCallback,
    weights: Dict[str, float],
) -> tuple[list[int], dict]:
    """Compute the final ranking from per-dimension verdicts and dimension weights.

    Scoring rule:
        - The winner of a dimension earns the full dimension weight.
        - The loser earns 0.
        - A tie (winner == 0) awards 0 to both skills.

    Returns:
        rank:     [rank_of_skill_1, rank_of_skill_2] — a permutation of [1, 2]
        scores:   {"skill_1": <weighted_total>, "skill_2": <weighted_total>}
    """
    score_1 = 0.0
    score_2 = 0.0

    for dim in _DIMENSIONS:
        comparison: DimensionComparison = getattr(parsed, f"{dim}_comparison")
        w = weights.get(dim, 0.)
        if comparison.winner == 1:
            score_1 += w
        elif comparison.winner == 2:
            score_2 += w
        # winner == 0 → tie, both get 0

    if score_1 >= score_2:
        rank = [1, 2]
    else:
        rank = [2, 1]

    return rank, {"skill_1": round(score_1, 4), "skill_2": round(score_2, 4)}


class SkillComprehensivePairwiseGrader(LLMGrader):
    """
    Skill Comprehensive Pairwise Grader

    Purpose:
        Compares exactly two AI Agent Skill packages against the same task description
        in a single LLM call.  The LLM evaluates each of the four quality dimensions —
        Relevance, Completeness, Safety, and Structure — and reports a head-to-head
        verdict (winner = 1 / 2 / 0 for tie) for each dimension.  The final ranking
        is then computed programmatically: the winner of each dimension earns that
        dimension's weight; the skill with the higher total weighted score is ranked 1st.

    Scoring mechanics:
        - Per-dimension: winner earns ``dimension_weights[dim]``, loser earns 0, tie → 0 each.
        - Total weighted score per skill = sum of earned dimension weights.
        - rank = [1, 2] if Skill 1 wins (or ties), [2, 1] if Skill 2 wins.
        - Weighted scores are exposed in ``result.metadata["weighted_scores"]``.

    What it evaluates:
        - Relevance:    which skill more directly addresses the specified task
        - Completeness: which skill provides more actionable, complete guidance
        - Safety:       which skill better avoids dangerous operations and scopes
                        permissions correctly
        - Structure:    which skill has a better NEVER list, description, content
                        layering, and freedom calibration

    When to use:
        - Selecting between two skill candidates before publishing to a registry
        - A/B testing two revisions of the same skill
        - Quick head-to-head audit of a community skill vs. an in-house skill
        - Final round comparison after filtering a larger pool with a listwise grader

    Args:
        model:             BaseChatModel instance or dict config for OpenAIChatModel
        dimension_weights: Per-dimension weights used for score aggregation.
                           Keys: "relevance", "completeness", "safety", "structure".
                           Missing keys default to 1.0.
                           (default: all dimensions equally weighted at 1.0)
        template:          Custom evaluation template
                           (default: DEFAULT_SKILL_COMPREHENSIVE_PAIRWISE_TEMPLATE)
        language:          Prompt language — EN or ZH (default: LanguageEnum.EN)
        strategy:          Evaluation strategy to use (default: DirectEvaluationStrategy)

    Returns:
        GraderRank with:
            - rank:     [1, 2] if Skill 1 wins overall, [2, 1] if Skill 2 wins overall
            - reason:   LLM-generated summary of key observed differences
            - metadata:
                - relevance_comparison:    {winner, reason}
                - completeness_comparison: {winner, reason}
                - safety_comparison:       {winner, reason}
                - structure_comparison:    {winner, reason}
                - weighted_scores:         {"skill_1": <float>, "skill_2": <float>}
                - dimension_weights:       {"relevance": ..., "completeness": ..., ...}

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.comprehensive_pairwise import (
        ...     SkillComprehensivePairwiseGrader,
        ... )
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillComprehensivePairwiseGrader(
        ...     model=model,
        ...     dimension_weights={"relevance": 2.0, "completeness": 1.5, "safety": 1.0, "structure": 1.0},
        ... )
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Review a pull request for code quality issues.",
        ...     skill_1={
        ...         "skill_name": "code-review",
        ...         "skill_description": "Structured code review for PRs. Use when reviewing diffs.",
        ...         "skill_md": "---\\nname: code-review\\n...\\n---\\n# NEVER\\n...",
        ...         "scripts": "",
        ...         "allowed_tools": "read_file",
        ...     },
        ...     skill_2={
        ...         "skill_name": "pr-summarizer",
        ...         "skill_description": "Summarizes pull requests. Use when generating PR descriptions.",
        ...         "skill_md": "---\\nname: pr-summarizer\\n...\\n---\\n",
        ...         "scripts": "",
        ...         "allowed_tools": "read_file",
        ...     },
        ... ))
        >>> print(result.rank)                              # e.g. [1, 2]
        >>> print(result.metadata["weighted_scores"])       # {"skill_1": 4.5, "skill_2": 1.0}
        >>> print(result.metadata["relevance_comparison"])  # {"winner": 1, "reason": "..."}
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_COMPREHENSIVE_PAIRWISE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        dimension_weights: Optional[Dict[str, float]] = None,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillComprehensivePairwiseGrader.

        Args:
            model:             BaseChatModel instance or dict config for OpenAIChatModel
            dimension_weights: Per-dimension weights for score aggregation.
                               Keys: "relevance", "completeness", "safety", "structure".
                               Missing keys default to 1.0.
            template:          PromptTemplate for evaluation prompts.
            language:          Language for prompts (default: LanguageEnum.EN).
            strategy:          The evaluation strategy to use.
        """
        super().__init__(
            name="skill_comprehensive_pairwise",
            mode=GraderMode.LISTWISE,
            description=(
                "Pairwise head-to-head comparison of two AI Agent Skills across "
                "relevance, completeness, safety, and structure"
            ),
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
            structured_model=SkillComprehensivePairwiseCallback,
        )
        self.dimension_weights: Dict[str, float] = {
            **DEFAULT_DIMENSION_WEIGHTS,
            **(dimension_weights or {}),
        }

    async def _aevaluate(
        self,
        task_description: str,
        skill_1: dict,
        skill_2: dict,
    ) -> GraderRank:
        """
        Compare two AI Agent Skill packages head-to-head across four quality dimensions.

        The LLM produces per-dimension verdicts only; the final ranking is computed
        here by weighting each dimension outcome and summing the scores.

        Args:
            task_description: Description of the task both skills should accomplish.
            skill_1: First skill dict. May contain:
                - skill_name (str):        Name of the skill
                - skill_description (str): Trigger/description text from skill metadata
                - skill_md (str):          Full content of the SKILL.md file
                - scripts (str):           Concatenated content of bundled scripts
                - allowed_tools (str):     Tools or permissions the skill may use
            skill_2: Second skill dict. Same keys as skill_1.

        Returns:
            GraderRank:
                rank     = [1, 2] if Skill 1 wins, [2, 1] if Skill 2 wins.
                reason   = LLM-generated summary of key differences.
                metadata = per-dimension comparisons + weighted_scores + dimension_weights.
        """
        try:
            # ── 1. Format prompt variables ──────────────────────────────────
            params = {
                **self.kwargs,
                "task_description": task_description,
                "skill_1_name": skill_1.get("skill_name", ""),
                "skill_1_description": skill_1.get("skill_description", ""),
                "skill_1_md": skill_1.get("skill_md", "") or "(none)",
                "skill_1_scripts": skill_1.get("scripts", "") or "(none)",
                "skill_1_allowed_tools": skill_1.get("allowed_tools", "") or "(none)",
                "skill_2_name": skill_2.get("skill_name", ""),
                "skill_2_description": skill_2.get("skill_description", ""),
                "skill_2_md": skill_2.get("skill_md", "") or "(none)",
                "skill_2_scripts": skill_2.get("scripts", "") or "(none)",
                "skill_2_allowed_tools": skill_2.get("allowed_tools", "") or "(none)",
            }

            # ── 2. Call the LLM ─────────────────────────────────────────────
            messages = self.template.format(language=self.language, **params)
            chat_response = await self.model.achat(
                messages=list(messages),
                structured_model=self.structured_model,
                callback=self.callback,
            )

            if hasattr(chat_response, "__aiter__"):
                async for chunk in chat_response:
                    chat_response = chunk

            raw = chat_response.parsed
            if isinstance(raw, dict):
                raw = SkillComprehensivePairwiseCallback(**raw)
            parsed: SkillComprehensivePairwiseCallback = raw

            # ── 3. Compute weighted ranking ─────────────────────────────────
            rank, weighted_scores = _compute_ranking(parsed, self.dimension_weights)

            # ── 4. Build metadata ───────────────────────────────────────────
            metadata = {
                "relevance_comparison": parsed.relevance_comparison.model_dump(),
                "completeness_comparison": parsed.completeness_comparison.model_dump(),
                "safety_comparison": parsed.safety_comparison.model_dump(),
                "structure_comparison": parsed.structure_comparison.model_dump(),
                "weighted_scores": weighted_scores,
                "dimension_weights": dict(self.dimension_weights),
            }

            return GraderRank(
                name=self.name,
                rank=rank,
                reason=parsed.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating skills pairwise: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "SkillComprehensivePairwiseGrader",
    "SkillComprehensivePairwiseCallback",
    "DimensionComparison",
    "DEFAULT_SKILL_COMPREHENSIVE_PAIRWISE_TEMPLATE",
    "DEFAULT_DIMENSION_WEIGHTS",
]
