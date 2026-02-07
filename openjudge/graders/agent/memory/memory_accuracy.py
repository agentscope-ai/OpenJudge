# -*- coding: utf-8 -*-
"""
Memory Accuracy Grader

Evaluates whether the agent stores accurate and factual information in its memory module.
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
MEMORY_ACCURACY_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent stores accurate and factual information in its memory module. This includes recording only information that was actually observed, storing correct interpretations as facts, and maintaining accurate details about objects or states.

<Rubrics>
1. The agent stores information in memory that accurately reflects what was present in the observation
2. The agent records only factual details (colors, quantities, locations) that were mentioned in observation
3. The agent saves correct interpretations of observations as factual memories
4. The agent creates accurate associations or relationships supported by observations
5. The agent's memory contains information that is consistent with and grounded in what was observed
</Rubrics>

<Steps>
1. Apply each rubric: Check if the step demonstrates good accuracy patterns described in each rubric
2. Focus on relevant modules: Only consider observation and memory modules
3. Provide evidence-based reasoning: Explain how the memory demonstrates accuracy and why
4. Assess confidence: Rate your confidence based on how clearly the accuracy is exhibited
</Steps>

<Scale>
- **Score 1.0**: The memory is accurate and factual (good accuracy)
- **Score 0.0**: The memory contains inaccuracies or fabrications (poor accuracy)
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
    "reason": "<detailed explanation of memory accuracy and confidence level>",
    "score": <0.0 or 1.0>
}}
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompt
MEMORY_ACCURACY_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体行为的专家。你的任务是评估智能体是否在其记忆模块中存储了准确且真实的信息。这包括只记录实际观察到的信息、将正确的解释存储为事实，以及维护关于对象或状态的准确细节。

<评分标准>
1. 智能体在记忆中存储的信息准确反映了观察中存在的内容
2. 智能体只记录了观察中提及的事实细节（颜色、数量、位置）
3. 智能体将对观察的正确解释保存为事实记忆
4. 智能体创建了观察支持的准确关联或关系
5. 智能体的记忆包含了与观察一致且基于观察的信息
</评分标准>

<评估步骤>
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好准确性模式
2. 关注相关模块：仅考虑观察和记忆模块
3. 提供基于证据的推理：解释记忆如何展示准确性以及原因
4. 评估置信度：根据准确性表现的清晰程度评估你的置信度
</评估步骤>

<评分量表>
- **分数 1.0**：记忆准确且真实（良好准确性）
- **分数 0.0**：记忆包含不准确或捏造的内容（准确性不佳）
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
    "reason": "<关于记忆准确性的详细解释和置信度水平>",
    "score": <0.0 或 1.0>
}}
</输出格式>
JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_MEMORY_ACCURACY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=MEMORY_ACCURACY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=MEMORY_ACCURACY_PROMPT_ZH,
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
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = MemoryAccuracyGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="You see a closed cabinet.",
        ...     memory="The cabinet is closed."
        ... ))
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_MEMORY_ACCURACY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize MemoryAccuracyGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts (default: DEFAULT_MEMORY_ACCURACY_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """

        super().__init__(
            name="memory_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluate memory accuracy",
            model=model,
            template=template or DEFAULT_MEMORY_ACCURACY_TEMPLATE,
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
        Evaluate memory accuracy

        Args:
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good accuracy, 0.0 = poor accuracy)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="You see a closed cabinet.",
            ...     memory="The cabinet is closed.",
            ...     context="Task: Inventory room objects"
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
