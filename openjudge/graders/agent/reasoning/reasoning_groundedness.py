# -*- coding: utf-8 -*-
"""
Reasoning Groundedness Grader

Evaluates whether the agent's reasoning/claims are grounded in actual observations
rather than speculation or unsupported assumptions.
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
REASONING_GROUNDEDNESS_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent reasoning. Your task is to evaluate whether the agent's reasoning is grounded in actual observations. Grounded reasoning means all claims, inferences, and conclusions are directly supported by evidence from the observation, without speculation, unsupported assumptions, or fabricated information.

<Rubrics>
1. The reasoning only makes claims that are directly supported by the observation
2. The reasoning does not introduce information not present in or derivable from the observation
3. Inferences in the reasoning follow logically from observation evidence
4. The reasoning does not make unsupported causal claims (e.g., assuming X caused Y without evidence)
5. The reasoning distinguishes between what was observed and what was inferred
6. The reasoning does not fill in gaps with assumed or hallucinated details
</Rubrics>

<Steps>
1. Extract claims: Identify all claims and inferences in the reasoning
2. Check evidence: For each claim, verify it is supported by the observation
3. Detect fabrication: Identify any information in the reasoning not present in the observation
4. Assess inference quality: Determine if inferences are logically justified
5. Evaluate overall groundedness: Are all reasoning steps properly supported?
</Steps>

<Scale>
- **Score 1.0**: The reasoning is well-grounded in observation (good groundedness)
- **Score 0.0**: The reasoning contains speculation or unsupported claims (poor groundedness)
</Scale>

<Context (Optional)>
{context}
</Context>

<History (Optional)>
{history}
</History>

<Current Step>
Observation: {observation}
Reasoning: {reasoning}
</Current Step>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of reasoning groundedness and confidence level>",
    "score": <0.0 or 1.0>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
REASONING_GROUNDEDNESS_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体推理的专家。你的任务是评估智能体的推理是否基于实际观察。基于事实的推理意味着所有声明、推断和结论都直接由观察中的证据支持，没有猜测、无根据的假设或编造的信息。

<评分标准>
1. 推理只做出直接由观察支持的声明
2. 推理没有引入观察中不存在或无法从中推导出的信息
3. 推理中的推断逻辑上遵循观察证据
4. 推理没有做出无根据的因果声明（例如，在没有证据的情况下假设X导致了Y）
5. 推理区分了观察到的内容和推断的内容
6. 推理没有用假设或幻觉的细节填补空白
</评分标准>

<评估步骤>
1. 提取声明：识别推理中的所有声明和推断
2. 检查证据：对于每个声明，验证它是否被观察支持
3. 检测虚构：识别推理中不在观察中的任何信息
4. 评估推断质量：确定推断是否有逻辑依据
5. 评估整体基于事实性：所有推理步骤是否都有适当支持？
</评估步骤>

<评分量表>
- **分数 1.0**：推理很好地基于观察（良好基于事实性）
- **分数 0.0**：推理包含猜测或无根据的声明（基于事实性不佳）
</评分量表>

<上下文（可选）>
{context}
</上下文>

<历史记录（可选）>
{history}
</历史记录>

<当前步骤>
观察：{observation}
推理：{reasoning}
</当前步骤>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于推理基于事实性的详细解释和置信度水平>",
    "score": <0.0 或 1.0>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=REASONING_GROUNDEDNESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=REASONING_GROUNDEDNESS_PROMPT_ZH,
            ),
        ],
    },
)


class ReasoningGroundednessGrader(LLMGrader):
    """
    Reasoning Groundedness Grader

    Evaluates whether the agent's reasoning/claims are grounded in actual observations.

    Required modules: observation, reasoning

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = ReasoningGroundednessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="The door is locked.",
        ...     reasoning="The door is locked, so I need to find a key."
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ReasoningGroundednessGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="reasoning_groundedness",
            mode=GraderMode.POINTWISE,
            description="Evaluate reasoning groundedness in observation",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        observation: str,
        reasoning: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reasoning groundedness.

        Args:
            observation: Agent's observation from the environment
            reasoning: Agent's reasoning (can be plan, reflection, or other reasoning text)
            history: Optional list of previous step dictionaries for context
            context: Optional task context

        Returns:
            GraderScore: Score with binary value (1.0 = grounded, 0.0 = ungrounded)
        """
        context_str = context if context else ""
        history_str = format_history(history, include_tags=False)

        try:
            result = await super()._aevaluate(
                observation=observation,
                reasoning=reasoning,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating reasoning groundedness: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "reasoning_groundedness",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReasoningGroundednessGrader",
    "DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE",
]
