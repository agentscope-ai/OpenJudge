# -*- coding: utf-8 -*-
"""
Skill Structure Grader

Evaluates whether an AI Agent Skill's internal structure is well-designed across four
dimensions: Anti-Pattern Quality, Specification Compliance, Progressive Disclosure,
and Freedom Calibration.
"""

import textwrap
from typing import Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
SKILL_STRUCTURE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI Skill architect. Your task is to assess whether an AI Agent Skill is structurally well-designed across four dimensions: Anti-Pattern Quality, Specification Compliance, Progressive Disclosure, and Freedom Calibration.

<Rubrics>
A structurally sound skill should satisfy all four dimensions:

**D1 — Anti-Pattern Quality**
Does the skill contain an effective NEVER list?
- Expert anti-patterns encode knowledge only experience teaches — each entry should state WHAT not to do and WHY in concrete, non-obvious terms. The test: would an expert say "yes, I learned this the hard way"? Or would they say "this is obvious to everyone"?
- High value: specific domain examples with non-obvious reasons ("NEVER use purple gradients because they signal AI-generated content and undermine design credibility"), decision-tree-style constraints, failure modes from real-world experience
- Low value / penalize: absent NEVER list; generic warnings that apply to any task ("be careful", "avoid errors", "handle edge cases") with no domain-specific reasoning; obvious statements anyone would know

**D2 — Specification Compliance (especially description)**
Does the skill follow official format requirements? The description is THE MOST CRITICAL field — it is the only thing the Agent reads before deciding to load the skill. A vague description renders the skill permanently invisible.

The skill activation flow:
  User Request → Agent sees ALL skill descriptions → Decides which to activate
                 (only descriptions, not bodies!)
  If description doesn't match → Skill NEVER gets loaded

- Valid `name`: lowercase, alphanumeric + hyphens only, ≤ 64 characters
- Description must answer THREE questions: WHAT it does (specific capabilities, not "handles X tasks"), WHEN to trigger it (explicit scenarios: "Use when...", "When user asks for..."), and KEYWORDS (file extensions, domain terms, action verbs that make it searchable)
- Penalize: description is vague ("handles document tasks", "a helpful skill for various tasks"); missing WHEN triggers; "When to use this Skill" guidance placed only in the body instead of the description field — the body is loaded AFTER the triggering decision is already made

**D3 — Progressive Disclosure**
Does the skill implement proper content layering?

Three loading layers:
  Layer 1 — Metadata (always in memory): only name + description (~100 tokens per skill)
  Layer 2 — SKILL.md body (loaded after triggering): detailed guidelines, decision trees — ideal < 500 lines, < 300 preferred
  Layer 3 — References (loaded on demand): scripts/, references/, assets/ — no size limit

- High value: MANDATORY loading triggers embedded at relevant workflow decision points (not just listed at the end); "Do NOT Load" guidance to prevent over-loading irrelevant files; SKILL.md stays focused as a routing/decision layer
- Low value / penalize: SKILL.md is a dump of all content (>500 lines, no offloading); references directory exists but files are never triggered (orphan references — knowledge present but never accessed); loading guidance only listed at the end without workflow integration
- For simple skills (<100 lines, no references): evaluate on conciseness and self-containment instead

**D4 — Freedom Calibration**
Is the constraint level appropriate for the task's fragility?

The freedom spectrum:
  Creative/design tasks     → High freedom:   principles, intent, trade-offs — NOT rigid step-by-step scripts
  Code review / analysis    → Medium freedom:  prioritized criteria, judgment-based ordering
  File format / irreversible → Low freedom:    exact scripts, precise parameters, explicit do-not-deviate instructions

- The test: "If the Agent makes a mistake, what is the consequence?" — high consequence → low freedom; low consequence → high freedom
- High value: constraint level calibrated per section to match each section's consequence of error
- Low value / penalize: rigid step-by-step scripts imposed on creative tasks (stifles valid variation and differentiation); vague high-level guidance given for operations where a wrong move causes data loss, file corruption, or security failure; uniform constraint level applied regardless of per-section fragility
</Rubrics>

<CommonFailurePatterns>
Watch for these patterns — each one indicates a specific dimension failure:
- The Vague Warning [D1]: "Be careful", "avoid errors", "consider edge cases" — NEVER list is absent or contains only generic statements
- The Invisible Skill [D2]: great content but description missing WHEN triggers and domain KEYWORDS
- The Wrong Location [D2]: "When to use this Skill" section placed in the body, not in the description field
- The Dump [D3]: SKILL.md is 500+ lines with everything included, no content offloading to references/
- The Orphan References [D3]: references/ directory exists but files are never loaded (no MANDATORY triggers embedded in workflow)
- The Freedom Mismatch [D4]: rigid scripts for creative tasks, or vague guidance for fragile/destructive operations
</CommonFailurePatterns>

<Steps>
1. Read the skill's name, description, and full SKILL.md content completely.
2. Check the NEVER list (D1): are anti-patterns specific, domain-relevant, and explained with non-obvious reasons? Would an expert recognize these as hard-won knowledge?
3. Check the description (D2): does it answer WHAT + WHEN + contain searchable KEYWORDS? Is any "when to use" guidance buried in the body instead?
4. Check content layering (D3): is SKILL.md appropriately sized? If references exist, are they loaded with MANDATORY triggers embedded in the workflow, not just listed?
5. Check freedom calibration (D4): for each section, does the constraint level match the consequence of error?
6. Note any common failure patterns detected.
7. Assign a score [1, 3] reflecting overall structural quality across all four dimensions.
8. Provide a concise reason citing specific evidence from the skill content.
</Steps>

<Constraints>
Base your evaluation strictly on the provided skill content; do not infer structure or intent that is not present.
If SKILL.md content is empty or missing, score is 1.
A score of 3 means the skill is structurally sound across all four dimensions with no significant gaps.
A score of 1 means the skill fails most structural criteria and would benefit from fundamental redesign.
</Constraints>

<Scale>
- 3: Structurally sound — expert-grade NEVER list with specific non-obvious domain reasoning; description fully answers WHAT + WHEN + contains searchable keywords; SKILL.md is properly sized with MANDATORY loading triggers embedded in workflow (or self-contained if simple); constraint level matches task fragility throughout with per-section calibration
- 2: Partially sound — passes on some structural dimensions but has notable gaps; e.g., NEVER list exists but is generic or partially specific, description lacks WHEN triggers or keywords, references are listed but never loaded via embedded triggers, or constraint level is mismatched in one or more sections
- 1: Structurally poor — fails most criteria; no NEVER list or only vague warnings; description too generic to trigger correctly; SKILL.md is an unstructured dump or references are orphaned; constraint level is severely mismatched for the task type
</Scale>

<Skill Name>
{skill_name}
</Skill Name>

<Skill Description>
{skill_description}
</Skill Description>

<SKILL.md Content>
{skill_md}
</SKILL.md Content>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<concise explanation citing concrete evidence from the skill content, covering: (1) anti-pattern quality — specific or vague?, (2) description completeness — WHAT/WHEN/KEYWORDS present?, (3) content layering — size and trigger quality, (4) freedom calibration — constraint level vs task fragility, and (5) any failure patterns detected>",
    "score": <integer 1, 2, or 3, where 3 = structurally sound and 1 = structurally poor>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SKILL_STRUCTURE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 架构师。你的任务是从以下四个结构维度评估 AI Agent Skill 的设计质量：反模式质量、规范合规性、渐进式披露、自由度校准。

<评分标准>
结构合理的 Skill 应满足全部四个维度：

**D1 — 反模式质量**
Skill 是否包含有效的 NEVER 列表？
- 专家级反模式传递的是只有实践才能积累的知识——每一条都应明确说明"不能做什么"以及"为什么"（具体的非直觉原因）。测试标准：专家看到这条是否会说"是的，我就是这样踩坑的"？还是"这谁都知道"？
- 高价值：附有非直觉原因的具体领域示例（"NEVER 使用紫色渐变，因为这是 AI 生成内容的标志，会损害设计可信度"）、来自实战经验的失败模式
- 低价值/扣分：NEVER 列表缺失；仅包含适用于任何任务的通用警告（"小心"、"避免错误"、"处理边界情况"），没有领域特有的具体原因；显而易见、人人皆知的表述

**D2 — 规范合规性（尤其是 description）**
Skill 是否遵循官方格式要求？description 是最关键的字段——这是 Agent 决定是否加载 Skill 前唯一读取的内容。description 模糊则 Skill 永远不会被激活。

Skill 激活流程：
  用户请求 → Agent 查看所有 Skill 的 description → 决定激活哪个
              （只看 description，不看正文！）
  description 不匹配 → Skill 永远不会被加载

- 有效的 `name`：小写字母数字 + 连字符，≤ 64 字符
- description 必须回答三个问题：WHAT（具体能做什么，而非"处理 X 相关功能"）、WHEN（明确的触发场景："使用时机……"、"当用户要求……"）、KEYWORDS（文件扩展名、领域术语、动作动词，使其可被检索）
- 扣分：description 模糊（"处理文档相关功能"、"适用于各种任务的 Skill"）；缺少 WHEN 触发词；"使用时机"信息只出现在正文而非 description 字段——正文是激活决策做出之后才加载的

**D3 — 渐进式披露**
Skill 是否实现了合理的内容分层？

三层加载架构：
  第1层 — 元数据（始终在内存中）：仅 name + description（每个 Skill 约100 token）
  第2层 — SKILL.md 正文（触发后加载）：详细指引、决策树——理想 < 500 行，建议 < 300 行
  第3层 — 参考资源（按需加载）：scripts/、references/、assets/——无大小限制

- 高价值：MANDATORY 加载触发器嵌入在工作流的相关决策节点（而非仅在末尾列出）；附有"Do NOT Load"指引防止无关文件被过度加载；SKILL.md 保持精简，作为路由/决策层
- 低价值/扣分：SKILL.md 堆砌所有内容（>500行，无内容卸载）；references 目录存在但文件从未被触发（孤立引用——知识存在但从未被访问）；加载指引仅在末尾列出，未集成到工作流
- 简单 Skill（<100行，无 references）：改为基于简洁性和自包含性进行评估

**D4 — 自由度校准**
约束程度是否与任务脆弱性相匹配？

自由度光谱：
  创意/设计任务       → 高自由度：原则、意图、权衡——而非刚性步骤脚本
  代码审查/分析       → 中等自由度：优先级标准，需要判断
  文件格式/不可逆操作  → 低自由度：精确脚本、明确参数、不得偏离的明确指令

- 测试方法："如果 Agent 在这里出错，后果是什么？"——后果严重 → 低自由度；后果轻微 → 高自由度
- 高价值：各章节的约束级别分别对应该章节的出错后果，而非全文统一约束
- 低价值/扣分：对创意任务强加刚性步骤脚本（压制合理变体和差异化）；对可能导致数据丢失、文件损坏或安全问题的操作只给出模糊的高层指引；全文使用统一约束级别而不考虑各章节脆弱性差异
</评分标准>

<常见失败模式>
识别以下模式——每种模式对应特定维度的失败：
- 模糊警告 [D1]："小心"、"避免错误"——NEVER 列表缺失或仅含通用表述
- 隐形 Skill [D2]：内容优质但 description 模糊，缺少 WHEN 触发词和领域 KEYWORDS
- 错误位置 [D2]："使用时机"放在正文而非 description 字段
- 堆砌模式 [D3]：SKILL.md 超过 500 行，包含所有内容，无内容卸载到 references/
- 孤立引用 [D3]：references/ 目录存在但文件从未被加载（工作流无嵌入的 MANDATORY 触发器）
- 自由度失配 [D4]：对创意任务的刚性脚本，或对脆弱/破坏性操作的模糊指引
</常见失败模式>

<评估步骤>
1. 完整阅读 Skill 的 name、description 和完整 SKILL.md 内容。
2. 检查 NEVER 列表（D1）：反模式是否具体、领域相关，且附有非直觉原因？专家会认可这些是实战积累的知识吗？
3. 检查 description（D2）：是否回答了 WHAT + WHEN + 包含可检索的关键词？是否有"使用时机"信息被埋在正文中而非 description 字段？
4. 检查内容分层（D3）：SKILL.md 是否大小合适？如果存在 references，是否通过嵌入工作流的 MANDATORY 触发器加载，而非仅列出？
5. 检查自由度校准（D4）：每个章节的约束级别是否与该章节的出错后果相匹配？
6. 记录检测到的常见失败模式。
7. 综合四个维度，给出 [1, 3] 的整体结构质量评分。
8. 提供简明理由，引用 Skill 内容的具体证据。
</评估步骤>

<注意事项>
严格基于提供的 Skill 内容进行评估，不要推断文本中未呈现的结构或意图。
如果 SKILL.md 内容为空或缺失，则评分为 1。
3 分表示 Skill 在全部四个结构维度上均合理，无明显缺口。
1 分表示 Skill 未能满足大多数结构标准，需要从根本上重新设计。
</注意事项>

<评分量表>
- 3：结构合理——专家级 NEVER 列表附有具体的非直觉领域原因；description 完整回答 WHAT + WHEN + 包含可检索的领域关键词；SKILL.md 大小合适，MANDATORY 加载触发器嵌入工作流（或简单 Skill 自包含）；约束级别与任务脆弱性逐章节匹配
- 2：部分合理——在部分结构维度上通过，但存在明显缺口；例如 NEVER 列表存在但过于通用或仅部分具体、description 缺少 WHEN 触发词或关键词、references 有列出但未通过嵌入触发器加载、一个或多个章节约束级别失配
- 1：结构较差——未能满足大多数标准；无 NEVER 列表或仅有模糊警告；description 过于通用无法正确触发；SKILL.md 是无结构的堆砌或存在孤立引用；任务类型与约束级别严重失配
</评分量表>

<Skill 名称>
{skill_name}
</Skill 名称>

<Skill 描述>
{skill_description}
</Skill 描述>

<SKILL.md 内容>
{skill_md}
</SKILL.md 内容>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<简明解释，引用 Skill 内容的具体证据，涵盖：（1）反模式质量——具体还是模糊？，（2）description 完整性——WHAT/WHEN/关键词是否齐全？，（3）内容分层——大小及触发器质量，（4）自由度校准——约束级别与任务脆弱性是否匹配，（5）检测到的失败模式>",
    "score": <整数 1、2 或 3，其中 3 = 结构合理，1 = 结构较差>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SKILL_STRUCTURE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SKILL_STRUCTURE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SKILL_STRUCTURE_PROMPT_ZH,
            ),
        ],
    },
)


class SkillStructureGrader(LLMGrader):
    """
    Skill Structure Grader

    Purpose:
        Evaluates whether an AI Agent Skill's internal structure is well-designed by assessing
        four structural dimensions derived from official Skill design specifications. Helps
        identify structural anti-patterns and improvement opportunities before deployment.

    What it evaluates:
        - Anti-Pattern Quality: Whether the skill contains specific, expert-grade NEVER lists
          with non-obvious domain reasons — not vague warnings like "be careful" or "avoid errors".
          The bar: would an expert recognize these as hard-won experience?
        - Specification Compliance: Whether frontmatter is valid and the description field
          answers WHAT/WHEN/KEYWORDS so the Agent can discover and trigger the skill correctly.
          The description is the only field read before the loading decision — vague = invisible.
        - Progressive Disclosure: Whether heavy content is offloaded to references/ with
          MANDATORY loading triggers embedded at workflow decision points (not just listed),
          keeping SKILL.md focused (< 500 lines, < 300 preferred)
        - Freedom Calibration: Whether the constraint level per section matches the task's
          fragility — high freedom (principles) for creative tasks, exact scripts for
          destructive/fragile operations, calibrated per section not uniformly applied

    When to use:
        - Auditing newly authored Skill packages before merging into a skill library
        - Automated CI checks on skill quality in a skills repository
        - Comparing competing skill designs for the same capability
        - Coaching skill authors on structural improvements

    Scoring (3-level scale):
        - 3 (Structurally sound): Expert-grade NEVER list with specific non-obvious domain
          reasoning; description fully answers WHAT/WHEN/KEYWORDS; SKILL.md properly sized
          with MANDATORY triggers embedded in workflow (or self-contained if simple); constraint
          level matches task fragility with per-section calibration
        - 2 (Partially sound): Passes some structural dimensions but has notable gaps; e.g.,
          NEVER list exists but is generic, description lacks WHEN triggers or keywords,
          references listed but not loaded via embedded triggers, or constraint mismatch
          in one or more sections
        - 1 (Structurally poor): Fails most criteria; no meaningful NEVER list; description
          too generic to trigger correctly; SKILL.md is an unstructured dump or references
          are orphaned; constraint level severely mismatched for the task type

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 3] to pass (default: 2)
        template: Custom evaluation template (default: DEFAULT_SKILL_STRUCTURE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Structure score [1, 3] where 3 = structurally sound, 1 = structurally poor
            - reason: Summary covering anti-pattern quality, description completeness,
                      content layering, freedom calibration, and detected failure patterns
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.structure import SkillStructureGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillStructureGrader(model=model, threshold=2)
        >>>
        >>> # Well-structured skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     skill_name="docx-editor",
        ...     skill_description=(
        ...         "Create, edit, and analyze .docx files including tracked changes, "
        ...         "comments, and formatting. Use when working with Word documents or "
        ...         "professional document formatting tasks."
        ...     ),
        ...     skill_md="---\\nname: docx-editor\\n...\\n---\\n# NEVER\\n..."
        ... ))
        >>> print(result.score)   # 3 - Structurally sound
        >>>
        >>> # Poorly structured skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     skill_name="helper",
        ...     skill_description="A helpful skill for various tasks.",
        ...     skill_md="# Helper\\nThis skill helps you do things. Be careful with errors.",
        ... ))
        >>> print(result.score)   # 1 - Structurally poor
        >>> print(result.reason)  # "No NEVER list; description too vague..."
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_STRUCTURE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 2,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillStructureGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum score [1, 3] to pass (default: 2)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_STRUCTURE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 3]
        """
        if not 1 <= threshold <= 3:
            raise ValueError(f"threshold must be in range [1, 3], got {threshold}")

        super().__init__(
            name="skill_structure",
            mode=GraderMode.POINTWISE,
            description="Evaluate structural quality of an AI Agent Skill across four dimensions: anti-pattern quality, specification compliance, progressive disclosure, and freedom calibration",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.threshold = threshold

    async def _aevaluate(
        self,
        skill_name: str,
        skill_description: str,
        skill_md: str = "",
    ) -> GraderScore:
        """
        Evaluate the structural quality of an AI Agent Skill.

        Args:
            skill_name: The name of the skill (e.g., "code-review")
            skill_description: The trigger/description text from the skill's frontmatter
            skill_md: Full content of the SKILL.md file. Defaults to empty string.

        Returns:
            GraderScore: Score in [1, 3] where:
                        3 = Structurally sound across all four dimensions,
                        2 = Partially sound with notable gaps in some dimensions,
                        1 = Structurally poor; fails most structural criteria.

        Example:
            >>> result = await grader.aevaluate(
            ...     skill_name="pdf-processor",
            ...     skill_description=(
            ...         "Extract text, tables, and metadata from PDF files. "
            ...         "Use when reading, summarising, or parsing .pdf documents."
            ...     ),
            ...     skill_md="---\\nname: pdf-processor\\n...\\n---\\n# NEVER\\n...",
            ... )
        """
        try:
            result = await super()._aevaluate(
                skill_name=skill_name,
                skill_description=skill_description,
                skill_md=skill_md or "(none)",
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={**result.metadata, "threshold": self.threshold},
            )

        except Exception as e:
            logger.exception(f"Error evaluating skill structure: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SkillStructureGrader", "DEFAULT_SKILL_STRUCTURE_TEMPLATE"]
