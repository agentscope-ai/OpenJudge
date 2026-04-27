# -*- coding: utf-8 -*-
"""
Reasoning Coherence Grader

Evaluates whether the agent's reasoning chain is logically coherent — each step
follows from the previous one without logical gaps, contradictions, or non-sequiturs.
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
REASONING_COHERENCE_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent reasoning. Your task is to evaluate whether the agent's reasoning chain is logically coherent. A coherent reasoning chain means each step follows logically from the previous one, without gaps, contradictions, or non-sequiturs.

<Rubrics>
1. The plan logically follows from the reflection (the plan addresses what the reflection identified)
2. The action logically follows from the plan (the action implements the plan's intent)
3. There are no contradictions between the plan and the action (e.g., plan says "search A" but action searches "B" without justification)
4. The reasoning chain shows clear causal progression (observation → reflection → plan → action)
5. There are no logical gaps where a step is missing between reasoning stages
</Rubrics>

<Steps>
1. Trace the reasoning chain: Follow the flow from reflection → plan → action
2. Check for logical continuity: Does each step naturally follow from the previous?
3. Identify contradictions: Are there any statements that conflict with each other?
4. Detect non-sequiturs: Are there any steps that don't follow from prior reasoning?
5. Assess overall coherence: Is the reasoning chain a connected logical whole?
</Steps>

<Scale>
- **Score 1.0**: The reasoning chain is logically coherent (good coherence)
- **Score 0.0**: The reasoning chain has logical gaps, contradictions, or non-sequiturs (poor coherence)
</Scale>

<Context (Optional)>
{context}
</Context>

<History (Optional)>
{history}
</History>

<Current Step>
Reflection: {reflection}
Plan: {plan}
Action: {action}
</Current Step>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of reasoning coherence and confidence level>",
    "score": <0.0 or 1.0>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
REASONING_COHERENCE_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体推理的专家。你的任务是评估智能体的推理链是否逻辑连贯。逻辑连贯的推理链意味着每个步骤都从上一步合乎逻辑地推出，没有逻辑漏洞、矛盾或无关跳跃。

<评分标准>
1. 计划合乎逻辑地源自反思（计划针对了反思识别出的问题）
2. 动作合乎逻辑地源自计划（动作实现了计划的意图）
3. 计划和动作之间没有矛盾（例如，计划说"搜索A"但动作搜索"B"且没有合理理由）
4. 推理链显示了清晰的因果递进（观察→反思→计划→动作）
5. 推理阶段之间没有逻辑漏洞（缺少必要的中间步骤）
</评分标准>

<评估步骤>
1. 追踪推理链：跟随反思→计划→动作的流程
2. 检查逻辑连续性：每个步骤是否自然地从前一步得出？
3. 识别矛盾：是否存在相互冲突的陈述？
4. 检测无关跳跃：是否有不遵循先前推理的步骤？
5. 评估整体连贯性：推理链是否是一个连贯的逻辑整体？
</评估步骤>

<评分量表>
- **分数 1.0**：推理链逻辑连贯（良好连贯性）
- **分数 0.0**：推理链存在逻辑漏洞、矛盾或无关跳跃（连贯性不佳）
</评分量表>

<上下文（可选）>
{context}
</上下文>

<历史记录（可选）>
{history}
</历史记录>

<当前步骤>
反思：{reflection}
计划：{plan}
动作：{action}
</当前步骤>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于推理连贯性的详细解释和置信度水平>",
    "score": <0.0 或 1.0>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_REASONING_COHERENCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=REASONING_COHERENCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=REASONING_COHERENCE_PROMPT_ZH,
            ),
        ],
    },
)


class ReasoningCoherenceGrader(LLMGrader):
    """
    Reasoning Coherence Grader

    Evaluates whether the agent's reasoning chain is logically coherent.

    Required modules: reflection, plan, action

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
        >>> grader = ReasoningCoherenceGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     reflection="The drawer is locked and I need a key.",
        ...     plan="I will search for the key in the cabinet.",
        ...     action="search cabinet"
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_REASONING_COHERENCE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ReasoningCoherenceGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="reasoning_coherence",
            mode=GraderMode.POINTWISE,
            description="Evaluate reasoning chain coherence",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        reflection: str,
        plan: str,
        action: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reasoning chain coherence.

        Args:
            reflection: Agent's reflection on the observation
            plan: Agent's planning/reasoning
            action: Agent's chosen action
            history: Optional list of previous step dictionaries for context
            context: Optional task context

        Returns:
            GraderScore: Score with binary value (1.0 = coherent, 0.0 = incoherent)
        """
        context_str = context if context else ""
        history_str = format_history(history, include_tags=False)

        try:
            result = await super()._aevaluate(
                reflection=reflection,
                plan=plan,
                action=action,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating reasoning coherence: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "reasoning_coherence",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReasoningCoherenceGrader",
    "DEFAULT_REASONING_COHERENCE_TEMPLATE",
]
