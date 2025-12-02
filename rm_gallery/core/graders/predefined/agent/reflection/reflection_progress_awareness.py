# -*- coding: utf-8 -*-
"""
Reflection Progress Awareness Grader

Evaluates whether the agent demonstrates accurate awareness of progress toward completing the task
in its reflection.
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
REFLECTION_PROGRESS_AWARENESS_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent demonstrates accurate awareness of progress toward completing the task in its reflection.

<Evaluation Type: Reflection Progress Awareness>
The agent should demonstrate accurate awareness of progress toward completing the task in its reflection. This includes correctly estimating progress and recognizing whether the agent is making forward progress or stuck in a loop.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent accurately assesses progress when making forward advancement
2. The agent correctly recognizes when it is stuck in a loop or repeating actions
3. The agent realistically estimates how close it is to task completion based on current state
4. The agent appropriately evaluates the progress made when observations show advancement
5. The agent demonstrates awareness of whether it is repeating failed attempts
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good progress awareness patterns described in each rubric
2. Focus on relevant modules: Only consider observation and reflection modules
3. Provide evidence-based reasoning: Explain how the reflection demonstrates progress awareness and why
4. Assess confidence: Rate your confidence based on how clearly the awareness is exhibited
</Evaluation Criteria>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# Scoring Instructions
- If progress awareness is accurate: score = 1.0 (good awareness)
- If progress awareness is inaccurate: score = 0.0 (poor awareness)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of progress awareness quality and confidence level>"
}}

JSON:
"""

# Chinese Prompt
REFLECTION_PROGRESS_AWARENESS_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是评估智能体是否在其反思中展示了对完成任务进度的准确意识。

<评估类型：反思进度意识>
智能体应该在其反思中展示对完成任务进度的准确意识。这包括正确估计进度并识别智能体是在向前推进还是陷入循环。
</评估类型>

<评估准则>
1. 智能体在向前推进时准确评估进度
2. 智能体在陷入循环或重复动作时正确识别
3. 智能体根据当前状态现实地估计距离任务完成的接近程度
4. 智能体在观察显示进展时适当评估所取得的进度
5. 智能体展示了对是否正在重复失败尝试的意识
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好进度意识模式
2. 关注相关模块：仅考虑观察和反思模块
3. 提供基于证据的推理：解释反思如何展示进度意识以及原因
4. 评估置信度：根据意识表现的清晰程度评估你的置信度
</评估标准>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# 评分指令
- 如果进度意识准确：score = 1.0（良好意识）
- 如果进度意识不准确：score = 0.0（意识不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于进度意识质量的详细解释和置信度水平>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_PROGRESS_AWARENESS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_PROGRESS_AWARENESS_PROMPT_ZH),
            ),
        ],
    },
)


class ReflectionProgressAwarenessGrader(LLMGrader):
    """
    Reflection Progress Awareness Grader

    Evaluates whether the agent demonstrates accurate awareness of progress toward completing the task
    in its reflection.

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
        >>> grader = ReflectionProgressAwarenessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="Cabinet 1 now has apples. Task complete.",
        ...     reflection="Good progress! I've successfully found the apples.",
        ...     task_context="Task: Find apples in cabinets"
        ... )
        >>> print(f"Score: {result.score}")  # 1.0 (accurate awareness)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reflection_progress_awareness",
            mode=GraderMode.POINTWISE,
            description="Evaluate reflection progress awareness",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE

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
        Evaluate reflection progress awareness

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good awareness, 0.0 = poor awareness)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="Cabinet 1 now has apples. Task complete.",
            ...     reflection="Good progress! I've successfully found the apples.",
            ...     task_context="Task: Find apples in cabinets"
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
            logger.error(f"Error evaluating reflection progress awareness: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "reflection_progress_awareness",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionProgressAwarenessGrader",
    "DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE",
]
