# -*- coding: utf-8 -*-
"""
Memory Accuracy Grader

Evaluates whether the agent stores accurate and factual information in its memory module.
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
MEMORY_ACCURACY_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent stores accurate and factual information in its memory module.

<Evaluation Type: Memory Accuracy>
The agent should store accurate and factual information in its memory module. This includes recording only information that was actually observed, storing correct interpretations as facts, and maintaining accurate details about objects or states.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent stores information in memory that accurately reflects what was present in the observation
2. The agent records only factual details (colors, quantities, locations) that were mentioned in observation
3. The agent saves correct interpretations of observations as factual memories
4. The agent creates accurate associations or relationships supported by observations
5. The agent's memory contains information that is consistent with and grounded in what was observed
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good accuracy patterns described in each rubric
2. Focus on relevant modules: Only consider observation and memory modules
3. Provide evidence-based reasoning: Explain how the memory demonstrates accuracy and why
4. Assess confidence: Rate your confidence based on how clearly the accuracy is exhibited
</Evaluation Criteria>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# Scoring Instructions
- If the memory is accurate and factual: score = 1.0 (good accuracy)
- If the memory contains inaccuracies or fabrications: score = 0.0 (poor accuracy)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of memory accuracy and confidence level>"
}}

JSON:
"""

# Chinese Prompt
MEMORY_ACCURACY_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是评估智能体是否在其记忆模块中存储了准确且真实的信息。

<评估类型：记忆准确性>
智能体应该在其记忆模块中存储准确且真实的信息。这包括只记录实际观察到的信息、将正确的解释存储为事实，以及维护关于对象或状态的准确细节。
</评估类型>

<评估准则>
1. 智能体在记忆中存储的信息准确反映了观察中存在的内容
2. 智能体只记录了观察中提及的事实细节（颜色、数量、位置）
3. 智能体将对观察的正确解释保存为事实记忆
4. 智能体创建了观察支持的准确关联或关系
5. 智能体的记忆包含了与观察一致且基于观察的信息
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好准确性模式
2. 关注相关模块：仅考虑观察和记忆模块
3. 提供基于证据的推理：解释记忆如何展示准确性以及原因
4. 评估置信度：根据准确性表现的清晰程度评估你的置信度
</评估标准>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# 评分指令
- 如果记忆准确且真实：score = 1.0（良好准确性）
- 如果记忆包含不准确或捏造的内容：score = 0.0（准确性不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于记忆准确性的详细解释和置信度水平>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_MEMORY_ACCURACY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_ACCURACY_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_ACCURACY_PROMPT_ZH),
            ),
        ],
    },
)


class MemoryAccuracyGrader(LLMGrader):
    """
    Memory Accuracy Grader

    Evaluates whether the agent stores accurate and factual information in its memory module.

    Required modules: observation, memory

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
        >>> grader = MemoryAccuracyGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="You see a closed cabinet.",
        ...     memory="The cabinet is closed."
        ... )
        >>> print(f"Score: {result.score}")  # 1.0 (good accuracy)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_MEMORY_ACCURACY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="memory_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluate memory accuracy",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_MEMORY_ACCURACY_TEMPLATE

    def _format_trajectory_steps(
        self,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.

        Args:
            observation: Agent's observation from the environment
            memory: Agent's memory content
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
        lines.append(f"Memory: {memory}")

        return "\n".join(lines)

    async def aevaluate(
        self,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate memory accuracy

        Args:
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good accuracy, 0.0 = poor accuracy)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="You see a closed cabinet.",
            ...     memory="The cabinet is closed.",
            ...     task_context="Task: Inventory room objects"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            observation=observation,
            memory=memory,
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
            logger.error(f"Error evaluating memory accuracy: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "memory_accuracy",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryAccuracyGrader",
    "DEFAULT_MEMORY_ACCURACY_TEMPLATE",
]
