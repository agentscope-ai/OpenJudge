# -*- coding: utf-8 -*-
"""
Reflection Outcome Understanding Grader

Evaluates whether the agent correctly understands and interprets the outcome or result of an action
in its reflection module.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent correctly understands and interprets the outcome or result of an action in its reflection.

<Evaluation Type: Reflection Outcome Understanding>
The agent should correctly understand and interpret the outcome or result of an action in its reflection. This occurs when the agent receives an observation indicating a specific result, and in the reflection module, the agent correctly understands what that observation means.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent correctly identifies whether an action succeeded or failed based on the observation
2. The agent accurately understands the state change that resulted from an action
3. The agent draws correct conclusions about what the observation indicates
4. The agent's reflection accurately captures the factual content of the observation
5. The agent demonstrates proper understanding of the action outcomes
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good understanding patterns described in each rubric
2. Focus on relevant modules: Only consider observation and reflection modules
3. Provide evidence-based reasoning: Explain how the reflection demonstrates understanding and why
4. Assess confidence: Rate your confidence based on how clearly the understanding is exhibited
</Evaluation Criteria>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# Scoring Instructions
- If the agent correctly understands the outcome: score = 1.0 (good understanding)
- If the agent misunderstands the outcome: score = 0.0 (poor understanding)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of outcome understanding quality and confidence level>"
}}

JSON:
"""

# Chinese Prompt
REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是评估智能体是否在其反思中正确理解和解释了动作的结果或输出。

<评估类型：反思结果理解>
智能体应该在其反思中正确理解和解释动作的结果或输出。这发生在智能体收到表明特定结果的观察时，并且在反思模块中，智能体正确理解了该观察的含义。
</评估类型>

<评估准则>
1. 智能体根据观察正确识别动作是成功还是失败
2. 智能体准确理解动作导致的状态变化
3. 智能体对观察所表明的内容得出了正确的结论
4. 智能体的反思准确捕捉了观察的事实内容
5. 智能体展示了对动作结果的正确理解
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好理解模式
2. 关注相关模块：仅考虑观察和反思模块
3. 提供基于证据的推理：解释反思如何展示理解以及原因
4. 评估置信度：根据理解表现的清晰程度评估你的置信度
</评估标准>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# 评分指令
- 如果智能体正确理解了结果：score = 1.0（良好理解）
- 如果智能体误解了结果：score = 0.0（理解不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于结果理解质量的详细解释和置信度水平>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_ZH),
            ),
        ],
    },
)


class ReflectionOutcomeUnderstandingGrader(LLMGrader):
    """
    Reflection Outcome Understanding Grader

    Evaluates whether the agent correctly understands and interprets the outcome or result of an action
    in its reflection module.

    Required modules: observation, reflection

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = ReflectionOutcomeUnderstandingGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="The drawer is now open.",
        ...     reflection="I successfully opened the drawer."
        ... )
        >>> print(f"Score: {result.score}")  # 1.0 (correct understanding)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reflection_outcome_understanding",
            mode=GraderMode.POINTWISE,
            description="Evaluate reflection outcome understanding",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE

    def _format_trajectory_steps(
        self,
        observation: str,
        reflection: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history_steps: Optional list of previous step dictionaries

        Returns:
            Formatted trajectory string
        """
        lines = []

        # Add history steps if provided
        if history_steps:
            for i, hist_step in enumerate(history_steps):
                lines.append(f"Step {i + 1}:")
                for key, value in hist_step.items():
                    if value:
                        lines.append(f"{key.capitalize()}: {value}")
                lines.append("")

        # Add current step
        step_number = len(history_steps) + 1 if history_steps else 1
        lines.append(f"Step {step_number}:")
        lines.append(f"Observation: {observation}")
        lines.append(f"Reflection: {reflection}")

        return "\n".join(lines)

    async def aevaluate(
        self,
        observation: str,
        reflection: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reflection outcome understanding

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good understanding, 0.0 = poor understanding)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="The drawer is now open.",
            ...     reflection="I successfully opened the drawer.",
            ...     task_context="Task: Open the drawer"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            observation=observation,
            reflection=reflection,
            history_steps=history_steps,
        )

        # Prepare context section
        context_section = ""
        if task_context:
            context_section = f"""<task_context>
{task_context}
</task_context>"""

        try:
            result = await super().aevaluate(
                trajectory_steps=trajectory_steps,
                context_section=context_section,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating reflection outcome understanding: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "reflection_outcome_understanding",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionOutcomeUnderstandingGrader",
    "DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE",
]
