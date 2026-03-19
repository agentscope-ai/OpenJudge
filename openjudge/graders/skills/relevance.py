# -*- coding: utf-8 -*-
"""
Skill Relevance Grader

Evaluates whether an AI Agent Skill's capabilities directly address a given task description.
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
SKILL_RELEVANCE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI skill evaluator. Your task is to assess how well an AI Agent Skill matches a given task description.

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

<Steps>
- Read the task description carefully to understand what capability or outcome is needed.
- Evaluate the skill's name, description, and SKILL.md content against the task.
- Assign a score [1, 3] based on how directly the skill addresses the task.
- Provide a concise reason citing concrete evidence from the skill content.
</Steps>

<Constraints>
Base your evaluation strictly on the provided skill content; do not infer capabilities that are not described.
A score of 3 means the skill directly and unambiguously addresses the task.
A score of 1 means the skill targets a different domain or task type entirely.
</Constraints>

<Scale>
- 3: Direct match — skill is explicitly designed for this task type; name, description, and SKILL.md clearly demonstrate it solves the task with little to no adaptation
- 2: Partial match — skill covers some aspects of the task but not all, or requires moderate domain adaptation; meaningful overlap but notable gaps remain
- 1: Poor match — skill targets a different domain or fundamentally different task type; applying it to this task would require substantial rework
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
    "score": <integer 1, 2, or 3, where 3 = direct match and 1 = poor match>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SKILL_RELEVANCE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 评估员。你的任务是评估 AI Agent Skill 与给定任务描述的匹配程度。

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

<评估步骤>
- 仔细阅读任务描述，了解所需的能力或成果。
- 对照任务评估 Skill 的名称、描述和 SKILL.md 内容。
- 根据 Skill 对任务的直接针对程度，给出评分 [1, 3]。
- 提供简明的理由，引用 Skill 内容中的具体证据。
</评估步骤>

<注意事项>
严格基于提供的 Skill 内容进行评估，不要推断未描述的能力。
3 分表示 Skill 直接且明确地针对该任务。
1 分表示 Skill 完全针对不同的领域或任务类型。
</注意事项>

<评分量表>
- 3：直接匹配——Skill 是专为此类任务设计的；名称、描述和 SKILL.md 内容清楚地表明它能以很少甚至不需要改动地解决该任务
- 2：部分匹配——Skill 涵盖了任务的某些方面但并非全部，或需要适度领域适配；存在有意义的功能重叠，但有明显差距
- 1：匹配较差——Skill 针对不同领域或完全不同类型的任务；将其用于给定任务需要大量重构
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
    "score": <整数 1、2 或 3，其中 3 = 直接匹配，1 = 匹配较差>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SKILL_RELEVANCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SKILL_RELEVANCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SKILL_RELEVANCE_PROMPT_ZH,
            ),
        ],
    },
)


class SkillRelevanceGrader(LLMGrader):
    """
    Skill Matching Grader

    Purpose:
        Evaluates whether an AI Agent Skill's described capabilities directly address
        a given task description. Useful for skill retrieval, routing, and gap analysis.

    What it evaluates:
        - Domain alignment: whether the skill targets the same domain as the task
        - Capability fit: whether the skill's features match what the task requires
        - Adaptation cost: how much rework would be needed to apply the skill to the task

    When to use:
        - Skill retrieval ranking: given a task, rank candidate skills by how well they match
        - Skill routing: select the most appropriate skill for an incoming user request
        - Skill gap analysis: detect tasks for which no existing skill provides a direct match
        - Recommender systems: surface the most applicable skill for a user request

    Scoring (3-level scale):
        - 3 (Direct match): Skill is explicitly designed for this task type; solves it with little to no adaptation
        - 2 (Partial match): Skill covers some aspects but requires moderate domain adaptation
        - 1 (Poor match): Skill targets a different domain or fundamentally different task type

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 3] to pass (default: 2)
        template: Custom evaluation template (default: DEFAULT_SKILL_RELEVANCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Matching score [1, 3] where 3 = direct match, 1 = poor match
            - reason: Explanation citing concrete evidence from the skill content
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.relevance import SkillRelevanceGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillRelevanceGrader(model=model, threshold=2)
        >>>
        >>> # Direct match
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Review a pull request for code quality issues.",
        ...     skill_name="code-review",
        ...     skill_description="Perform code reviews on pull requests, checking for bugs and style.",
        ...     skill_md="# Code Review\\n## Steps\\n1. Fetch PR diff\\n2. Analyze for bugs...",
        ... ))
        >>> print(result.score)   # 3 - Direct match
        >>>
        >>> # Poor match
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Generate a financial report from CSV data.",
        ...     skill_name="code-review",
        ...     skill_description="Perform code reviews on pull requests.",
        ...     skill_md="# Code Review\\nReview code diffs for quality issues.",
        ... ))
        >>> print(result.score)   # 1 - Poor match
        >>> print(result.reason)  # "Skill is designed for code review, not financial reporting."
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_RELEVANCE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 2,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillRelevanceGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum score [1, 3] to pass (default: 2)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_RELEVANCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 3]
        """
        if not 1 <= threshold <= 3:
            raise ValueError(f"threshold must be in range [1, 3], got {threshold}")

        super().__init__(
            name="skill_relevance",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether an AI Agent Skill is relevant to the task requirements",
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
        Evaluate how well an AI Agent Skill matches a given task description.

        Args:
            task_description: Description of the task the skill should accomplish
            skill_name: Name of the skill (e.g., "code-review")
            skill_description: The trigger/description text from the skill metadata
            skill_md: Full content of the SKILL.md file. Defaults to empty string.

        Returns:
            GraderScore: Score in [1, 3] where:
                        3 = Direct match (skill explicitly designed for this task type),
                        2 = Partial match (covers some aspects, requires adaptation),
                        1 = Poor match (different domain or fundamentally different task)

        Example:
            >>> result = await grader.aevaluate(
            ...     task_description="Summarize a PDF document.",
            ...     skill_name="pdf-summarizer",
            ...     skill_description="Extracts and summarizes PDF documents up to 20 pages.",
            ...     skill_md="# PDF Summarizer\\n## Steps\\n1. Load PDF\\n2. Summarize.",
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
            logger.exception(f"Error evaluating skill relevance: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SkillRelevanceGrader", "DEFAULT_SKILL_RELEVANCE_TEMPLATE"]

