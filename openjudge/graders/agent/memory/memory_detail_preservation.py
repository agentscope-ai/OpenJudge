# -*- coding: utf-8 -*-
"""
Memory Detail Preservation Grader

Evaluates whether the agent preserves important details when storing information in memory.
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
MEMORY_DETAIL_PRESERVATION_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent preserves important details when storing information in memory. This includes maintaining critical information such as exact locations, specific values, and important constraints, making the stored memory useful and actionable for future decision-making.

<Rubrics>
1. The agent stores specific details when the observation contained them
2. The agent preserves exact locations, coordinates, or spatial information present in observation
3. The agent maintains specific numerical values (quantities, distances, measurements) when storing
4. The agent preserves important constraints, conditions, or qualifiers from observed information
5. The agent's memory is sufficiently detailed and actionable based on the observation
</Rubrics>

<Steps>
1. Apply each rubric: Check if the step demonstrates good detail preservation patterns described in each rubric
2. Focus on relevant modules: Only consider observation and memory modules
3. Provide evidence-based reasoning: Explain how the memory preserves details and why
4. Assess confidence: Rate your confidence based on how clearly the preservation is exhibited
</Steps>

<Scale>
- **Score 1.0**: Important details are preserved (good detail preservation)
- **Score 0.0**: Important details are lost (poor detail preservation)
</Scale>

<Context (Optional)>
{context}
</Context>

<History (Optional)>
{history}
</History>

<Current Step>
Observation: {observation}
Memory: {memory}
</Current Step>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of detail preservation quality and confidence level>",
    "score": <0.0 or 1.0>
}}
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompt
MEMORY_DETAIL_PRESERVATION_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体行为的专家。你的任务是评估智能体在将信息存储到记忆中时是否保留了重要细节。这包括维护关键信息，如确切位置、具体数值和重要约束，使得存储的记忆对未来的决策有用且可操作。

<评分标准>
1. 智能体在观察包含具体细节时存储了它们
2. 智能体保留了观察中存在的确切位置、坐标或空间信息
3. 智能体在存储时维护了具体的数值（数量、距离、测量值）
4. 智能体保留了观察到的信息中的重要约束、条件或限定词
5. 基于观察，智能体的记忆足够详细且可操作
</评分标准>

<评估步骤>
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好细节保留模式
2. 关注相关模块：仅考虑观察和记忆模块
3. 提供基于证据的推理：解释记忆如何保留细节以及原因
4. 评估置信度：根据保留表现的清晰程度评估你的置信度
</评估步骤>

<评分量表>
- **分数 1.0**：重要细节被保留（良好的细节保留）
- **分数 0.0**：重要细节丢失（细节保留不佳）
</评分量表>

<上下文（可选）>
{context}
</上下文>

<历史记录（可选）>
{history}
</历史记录>

<当前步骤>
观察：{observation}
记忆：{memory}
</当前步骤>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于细节保留质量的详细解释和置信度水平>",
    "score": <0.0 或 1.0>
}}
</输出格式>
JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_MEMORY_DETAIL_PRESERVATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=MEMORY_DETAIL_PRESERVATION_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=MEMORY_DETAIL_PRESERVATION_PROMPT_ZH,
            ),
        ],
    },
)


class MemoryDetailPreservationGrader(LLMGrader):
    """
    Memory Detail Preservation Grader

    Evaluates whether the agent preserves important details when storing information in memory.

    Required modules: observation, memory

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
        >>> grader = MemoryDetailPreservationGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
        ...     memory="Cabinet 1 at (3.5, 2.1) has 5 red apples."
        ... ))
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_MEMORY_DETAIL_PRESERVATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize MemoryDetailPreservationGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [1, 5] (default: 3)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_MEMORY_DETAIL_PRESERVATION_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="memory_detail_preservation",
            mode=GraderMode.POINTWISE,
            description="Evaluate memory detail preservation",
            model=model,
            template=template or DEFAULT_MEMORY_DETAIL_PRESERVATION_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        observation: str,
        memory: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate memory detail preservation

        Args:
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good preservation, 0.0 = poor preservation)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
            ...     memory="Cabinet 1 at (3.5, 2.1) has 5 red apples.",
            ...     context="Task: Inventory items with precise locations"
            ... )
        """
        # Format context section
        context_str = context if context else ""

        # Format history
        history_str = format_history(history, include_tags=False)

        try:
            result = await super()._aevaluate(
                observation=observation,
                memory=memory,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating memory detail preservation: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "memory_detail_preservation",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryDetailPreservationGrader",
    "DEFAULT_MEMORY_DETAIL_PRESERVATION_TEMPLATE",
]
