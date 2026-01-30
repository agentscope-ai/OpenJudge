# -*- coding: utf-8 -*-
"""
Reflection Accuracy Grader

Evaluates whether the agent provides accurate reflections based on actual observations.
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.agent.utils import format_history
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
REFLECTION_ACCURACY_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent provides accurate reflections based on actual observations.

<Evaluation Type: Reflection Accuracy>
The agent should provide accurate reflections based on actual observations. The agent should reflect only on objects, states, and details that are actually mentioned in the observation, without inventing or fabricating information.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent mentions only objects or entities in reflection that were present in the observation
2. The agent describes only states or conditions that were reported in the observation
3. The agent uses only specific details (colors, numbers, locations) found in the observation
4. The agent reflects only on things that were actually seen or detected in the observation
5. The agent's reflection includes only information that can be derived from the observation
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good accuracy patterns described in each rubric
2. Focus on relevant modules: Only consider observation and reflection modules
3. Provide evidence-based reasoning: Explain how the reflection demonstrates accuracy and why
4. Assess confidence: Rate your confidence based on how clearly the accuracy is exhibited
</Evaluation Criteria>

{context}

{history}

<Current Step>
Observation: {observation}
Reflection: {reflection}
</Current Step>

# Scoring Instructions
- If the reflection is accurate and grounded: score = 1.0 (good accuracy)
- If the reflection contains fabrications: score = 0.0 (poor accuracy)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of reflection accuracy and confidence level>"
}}

JSON:
"""
).strip()

# Chinese Prompt
REFLECTION_ACCURACY_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析智能体行为的专家。你的任务是评估智能体是否基于实际观察提供了准确的反思。

<评估类型：反思准确性>
智能体应该基于实际观察提供准确的反思。智能体应该只反思实际在观察中提到的对象、状态和细节，而不捏造或编造信息。
</评估类型>

<评估准则>
1. 智能体在反思中只提到了观察中存在的对象或实体
2. 智能体只描述了观察中报告的状态或条件
3. 智能体只使用了观察中找到的具体细节（颜色、数字、位置）
4. 智能体只反思在观察中实际看到或检测到的内容
5. 智能体的反思只包含了可以从观察中推导出的信息
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好准确性模式
2. 关注相关模块：仅考虑观察和反思模块
3. 提供基于证据的推理：解释反思如何展示准确性以及原因
4. 评估置信度：根据准确性表现的清晰程度评估你的置信度
</评估标准>

{context}

{history}

<当前步骤>
观察：{observation}
反思：{reflection}
</当前步骤>

# 评分指令
- 如果反思准确且基于事实：score = 1.0（良好准确性）
- 如果反思包含捏造的内容：score = 0.0（准确性不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于反思准确性的详细解释和置信度水平>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_REFLECTION_ACCURACY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=REFLECTION_ACCURACY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=REFLECTION_ACCURACY_PROMPT_ZH,
            ),
        ],
    },
)


class ReflectionAccuracyGrader(LLMGrader):
    """
    Reflection Accuracy Grader

    Evaluates whether the agent provides accurate reflections based on actual observations.

    Required modules: observation, reflection

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = ReflectionAccuracyGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="You see a closed cabinet.",
        ...     reflection="I observed a closed cabinet."
        ... ))
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_REFLECTION_ACCURACY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ReflectionAccuracyGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts (default: DEFAULT_REFLECTION_ACCURACY_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="reflection_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluate reflection accuracy",
            model=model,
            template=template or DEFAULT_REFLECTION_ACCURACY_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        observation: str,
        reflection: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reflection accuracy

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = accurate, 0.0 = inaccurate)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="You see a closed cabinet.",
            ...     reflection="I observed a closed cabinet.",
            ...     context="Task: Find objects in the room"
            ... )
        """
        # Format context section
        context_str = f"<context>\n{context}\n</context>" if context else ""

        # Format history
        history_str = format_history(history)

        try:
            result = await super()._aevaluate(
                observation=observation,
                reflection=reflection,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating reflection accuracy: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "reflection_accuracy",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionAccuracyGrader",
    "DEFAULT_REFLECTION_ACCURACY_TEMPLATE",
]
