# -*- coding: utf-8 -*-
"""
Skill Completeness Grader

Evaluates whether an AI Agent Skill provides sufficient steps, inputs/outputs, prerequisites,
and error-handling guidance to accomplish a given task.
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
SKILL_COMPLETENESS_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI skill evaluator. Your task is to assess whether an AI Agent Skill provides sufficient detail to accomplish a given task.

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

<Steps>
- Read the task description to understand what a complete solution requires.
- Carefully examine the skill's SKILL.md content, checking steps, inputs, outputs, and prerequisites.
- Audit any formulas, algorithms, or code snippets line-by-line for correctness.
- Assign a score [1, 3] based on how thoroughly the skill covers what is needed.
- Provide a concise reason citing concrete evidence from the skill content.
</Steps>

<Constraints>
Base your evaluation strictly on the provided skill content; do not infer steps or details that are not described.
If SKILL.md content is empty or missing, this reflects a score of 1.
A score of 3 means the skill is actionable as written with no significant gaps.
A score of 1 means the skill is too vague to act on or contains critical errors.
</Constraints>

<Scale>
- 3: Complete — clear goal with explicit steps, inputs, and outputs; prerequisites mentioned when relevant; failure modes or edge cases addressed at least briefly
- 2: Partially complete — goal is clear but steps, prerequisites, or outputs are underspecified; assumes context the user may not have; missing some critical detail
- 1: Incomplete — too vague to act on; missing core steps; unclear what "done" looks like; or promises significant capabilities that the implementation does not actually provide
</Scale>

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

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<brief explanation citing concrete evidence from the skill content for the assigned score>",
    "score": <integer 1, 2, or 3, where 3 = complete and 1 = incomplete>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SKILL_COMPLETENESS_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 评估员。你的任务是评估 AI Agent Skill 是否提供了完成给定任务所需的充足细节。

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

<评估步骤>
- 阅读任务描述，了解完整的解决方案需要哪些内容。
- 仔细检查 Skill 的 SKILL.md 内容，核查步骤、输入、输出和前置条件。
- 逐行审计所有公式、算法或代码片段的正确性。
- 根据 Skill 对所需内容的覆盖程度，给出评分 [1, 3]。
- 提供简明的理由，引用 Skill 内容中的具体证据。
</评估步骤>

<注意事项>
严格基于提供的 Skill 内容进行评估，不要推断未描述的步骤或细节。
如果 SKILL.md 内容为空或缺失，则评分为 1。
3 分表示 Skill 按照现有内容即可操作，没有明显缺口。
1 分表示 Skill 过于模糊无法操作，或包含关键错误。
</注意事项>

<评分量表>
- 3：完整——目标明确，步骤、输入和输出清晰；在相关时提及前置条件；至少简要说明失败模式或边界情况
- 2：部分完整——目标清晰，但步骤、前置条件或输出规范不足；假设了用户可能没有的上下文；缺少某些关键细节
- 1：不完整——过于模糊，无法据此操作；缺少核心步骤；不清楚"完成"是什么样子；或承诺了重要功能但实现并未真正提供
</评分量表>

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

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<对所给分数的简要解释，引用 Skill 内容中的具体证据>",
    "score": <整数 1、2 或 3，其中 3 = 完整，1 = 不完整>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SKILL_COMPLETENESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SKILL_COMPLETENESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SKILL_COMPLETENESS_PROMPT_ZH,
            ),
        ],
    },
)


class SkillCompletenessGrader(LLMGrader):
    """
    Skill Completeness Grader

    Purpose:
        Evaluates whether an AI Agent Skill provides sufficient steps, inputs/outputs,
        prerequisites, and error-handling guidance to accomplish a given task.
        Detects vague or placeholder implementations that cannot reliably deliver
        on the skill's stated capabilities.

    What it evaluates:
        - Step coverage: whether the skill describes concrete, actionable steps
        - Input/output specification: whether inputs, outputs, and expected results are clear
        - Prerequisites: whether required environment, packages, or permissions are mentioned
        - Edge cases: whether failure modes or boundary conditions are addressed
        - Implementation quality: whether code/formulas in the skill are correct and non-trivial

    When to use:
        - Skill quality gating: ensure new skills are complete before publication
        - Skill auditing: identify existing skills that need improvement
        - Evaluating skill generation: measure how well auto-generated skills cover a task
        - Debugging failed executions: check if incomplete skill instructions caused the failure

    Scoring (3-level scale):
        - 3 (Complete): Clear goal with explicit steps, inputs/outputs, prerequisites, and edge cases
        - 2 (Partially complete): Goal is clear but steps/prereqs are underspecified or assume unstated context
        - 1 (Incomplete): Too vague to act on, missing core steps, or placeholder implementation

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 3] to pass (default: 2)
        template: Custom evaluation template (default: DEFAULT_SKILL_COMPLETENESS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Completeness score [1, 3] where 3 = complete, 1 = incomplete
            - reason: Explanation citing concrete evidence from the skill content
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.completeness import SkillCompletenessGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillCompletenessGrader(model=model, threshold=2)
        >>>
        >>> # Complete skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Summarize a PDF document.",
        ...     skill_name="pdf-summarizer",
        ...     skill_description="Extracts and summarizes PDF documents up to 20 pages.",
        ...     skill_md="# PDF Summarizer\\n## Prerequisites\\npip install pdfplumber\\n"
        ...               "## Steps\\n1. Load PDF\\n2. Chunk by paragraph\\n3. Summarize each chunk.",
        ... ))
        >>> print(result.score)   # 3 - Complete
        >>>
        >>> # Incomplete skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Summarize a PDF document.",
        ...     skill_name="pdf-summarizer",
        ...     skill_description="Summarizes PDFs.",
        ...     skill_md="# PDF Summarizer\\nLoad the file and summarize it.",
        ... ))
        >>> print(result.score)   # 1 - Incomplete
        >>> print(result.reason)  # "No steps, prerequisites, or output format are specified..."
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_COMPLETENESS_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 2,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillCompletenessGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum score [1, 3] to pass (default: 2)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_COMPLETENESS_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 3]
        """
        if not 1 <= threshold <= 3:
            raise ValueError(f"threshold must be in range [1, 3], got {threshold}")

        super().__init__(
            name="skill_completeness",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether an AI Agent Skill provides sufficient detail to accomplish a task",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.threshold = threshold

    async def _aevaluate(
        self,
        task_description: str,
        skill_name: str,
        skill_description: str,
        skill_md: str = "",
    ) -> GraderScore:
        """
        Evaluate whether an AI Agent Skill provides sufficient detail to accomplish a task.

        Args:
            task_description: Description of the task the skill should accomplish
            skill_name: Name of the skill (e.g., "pdf-summarizer")
            skill_description: The trigger/description text from the skill metadata
            skill_md: Full content of the SKILL.md file. Defaults to empty string.

        Returns:
            GraderScore: Score in [1, 3] where:
                        3 = Complete (clear goal, explicit steps, inputs/outputs, edge cases),
                        2 = Partially complete (goal clear but steps/prereqs underspecified),
                        1 = Incomplete (vague, missing core steps, or placeholder implementation)

        Example:
            >>> result = await grader.aevaluate(
            ...     task_description="Review a pull request for security vulnerabilities.",
            ...     skill_name="security-code-review",
            ...     skill_description="Reviews code for OWASP Top 10 vulnerabilities.",
            ...     skill_md="# Security Review\\n## Steps\\n1. Fetch diff\\n2. Check for SQLi...",
            ... )
        """
        try:
            result = await super()._aevaluate(
                task_description=task_description,
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
            logger.exception(f"Error evaluating skill completeness: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SkillCompletenessGrader", "DEFAULT_SKILL_COMPLETENESS_TEMPLATE"]
