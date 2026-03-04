# -*- coding: utf-8 -*-
"""
Proactive Interaction Grader for Multi-turn Conversations (PI)

Evaluates whether the assistant can proactively engage in conversation,
maintain dialogue rhythm, and guide the conversation forward naturally.

This is an advanced interaction ability that tests the model's capacity to:
- Proactively introduce relevant topics or questions
- Maintain natural conversation flow and rhythm
- Show genuine interest and engagement
- Guide users toward productive dialogue outcomes
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.utils.utils import format_conversation_history

# ============================================================================
# Prompt Templates
# ============================================================================

PROACTIVE_INTERACTION_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的主动互动能力评估专家。您的任务是评估多轮对话中助手是否能够主动抛出话题、维持对话节奏，展现出积极参与对话的能力。

<评分标准>
完美的主动互动应该：
- 在回答用户问题后，能够自然地提出相关的后续问题或话题。
- 展现出对用户需求的关注和兴趣。
- 主动提供有价值的补充信息或建议。
- 维持对话的流畅性和连贯性，避免对话陷入僵局。
- 引导对话向有意义的方向发展。
</评分标准>

<评估步骤>
1. 分析对话历史，了解当前对话的主题和用户的需求。
2. 评估助手的回复是否仅仅是被动回答，还是有主动互动的元素。
3. 检查助手是否提出了相关的后续问题或建议。
4. 评估这些主动互动是否自然、恰当，不显得突兀或过度。
5. 判断主动互动是否有助于推进对话或满足用户需求。
</评估步骤>

<注意事项>
主动互动应该自然而不强迫。过度追问或不相关的话题延伸应该扣分。如果用户的问题已经得到完整回答且不需要进一步互动，简洁的回复也是可以接受的。
</注意事项>

<评分量表>
- 5分：完美互动，助手自然地主动提出相关问题或话题，有效推进对话，展现出真诚的参与感。
- 4分：良好互动，助手有主动互动的尝试，基本自然恰当，但可能略显生硬或不够深入。
- 3分：基本互动，助手有一些主动元素，但互动不够自然或相关性不强。
- 2分：互动不足，助手主要是被动回答，缺乏主动推进对话的意识。
- 1分：无互动，助手完全被动，回复机械，没有任何主动参与对话的表现。
</评分量表>

<对话历史>
{history}
</对话历史>

<当前助手回复>
{response}
</当前助手回复>

<输出格式>
请按以下JSON格式输出评估结果：
{{
    "reason": "详细说明评估理由，包括主动互动的质量和自然程度分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


PROACTIVE_INTERACTION_PROMPT_EN = textwrap.dedent(
    """
You are a professional Proactive Interaction Evaluation Expert. Your task is to
evaluate whether the assistant can proactively engage in conversation, maintain
dialogue rhythm, and demonstrate active participation.

<Rubrics>
Perfect proactive interaction should:
- Naturally propose relevant follow-up questions or topics after answering user questions.
- Show genuine interest and attention to user needs.
- Proactively provide valuable supplementary information or suggestions.
- Maintain conversation fluency and coherence, avoiding dialogue deadlocks.
- Guide the conversation toward meaningful directions.
</Rubrics>

<Steps>
1. Analyze conversation history to understand the current topic and user needs.
2. Evaluate whether the assistant's response is merely passive answering or contains proactive elements.
3. Check if the assistant proposed relevant follow-up questions or suggestions.
4. Evaluate if these proactive interactions are natural and appropriate, not abrupt or excessive.
5. Determine if the proactive interaction helps advance the conversation or meet user needs.
</Steps>

<Constraints>
Proactive interaction should be natural, not forced. Excessive questioning or
irrelevant topic extensions should be penalized. If the user's question has been fully
answered and no further interaction is needed, a concise response is acceptable.
</Constraints>

<Scale>
- 5: Perfect interaction - naturally proposed relevant questions or topics, effectively
  advanced dialogue, showed genuine engagement.
- 4: Good interaction - attempted proactive interaction, mostly natural and appropriate,
  but slightly stiff or not deep enough.
- 3: Basic interaction - some proactive elements, but interaction not natural enough or relevance is weak.
- 2: Insufficient interaction - mainly passive answering, lacking awareness to proactively advance dialogue.
- 1: No interaction - completely passive, mechanical responses, no proactive participation in conversation.
</Scale>

<Conversation History>
{history}
</Conversation History>

<Current Assistant Response>
{response}
</Current Assistant Response>

<Output Schema>
Output your evaluation in the following JSON format:
{{
    "reason": "detailed explanation including analysis of proactive interaction quality and naturalness",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_PROACTIVE_INTERACTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=PROACTIVE_INTERACTION_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=PROACTIVE_INTERACTION_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class ProactiveInteractionGrader(LLMGrader):
    """
    Proactive Interaction Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can proactively engage in conversation,
        maintain dialogue rhythm, and guide the conversation forward naturally,
        creating meaningful and productive interactions.

    What it evaluates:
        - Follow-up Questions: Naturally proposing relevant questions after answering
        - User Engagement: Showing genuine interest in user needs
        - Value Addition: Proactively providing supplementary information
        - Conversation Flow: Maintaining fluency and avoiding deadlocks
        - Dialogue Guidance: Steering conversation toward meaningful directions

    When to use:
        - Evaluating conversational engagement quality
        - Assessing chatbot interaction naturalness
        - Testing dialogue management capabilities
        - Quality assurance for customer service bots
        - Measuring conversational AI proactiveness

    Scoring:
        - 5: Perfect interaction - natural, relevant, and effective engagement
        - 4: Good interaction - mostly natural with minor stiffness
        - 3: Basic interaction - some proactive elements but not natural
        - 2: Insufficient interaction - mainly passive responses
        - 1: No interaction - completely passive and mechanical

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_PROACTIVE_INTERACTION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect interaction, 1 = no interaction
            - reason: Explanation of interaction quality analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import ProactiveInteractionGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = ProactiveInteractionGrader(model=model)
        >>>
        >>> # Evaluate proactive interaction (response as string)
        >>> history = [
        ...     {"role": "user", "content": "I'm planning a trip to Japan."},
        ... ]
        >>> response = "Japan is wonderful! When are you planning to go? I can suggest activities based on the season."
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: high score for proactive engagement
        >>>
        >>> # Evaluate proactive interaction (response as dict)
        >>> response_dict = {"role": "assistant", "content": "Japan is wonderful! When are you planning to go?"}
        >>> result = asyncio.run(grader.aevaluate(response=response_dict, history=history))
        >>> print(result.score)  # Expected: high score for proactive engagement
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize ProactiveInteractionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_PROACTIVE_INTERACTION_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="proactive_interaction",
            mode=GraderMode.POINTWISE,
            description="Evaluate proactive interaction ability in multi-turn conversations",
            template=template or DEFAULT_PROACTIVE_INTERACTION_TEMPLATE,
            language=language,
            strategy=strategy,
            **kwargs,
        )

    async def _aevaluate(
        self,
        response: str | Dict[str, Any],
        history: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate proactive interaction of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 proactive interaction score
                - reason: Detailed evaluation explanation
                - metadata: Contains evaluation type and history info
        """
        try:
            # Handle response as string or dict
            if isinstance(response, dict):
                response_str = response.get("content", str(response))
            else:
                response_str = response

            history_str = format_conversation_history(history)

            result = await super()._aevaluate(
                response=response_str,
                history=history_str,
                **kwargs,
            )

            metadata = {
                "history_turns": len(history),
                "evaluation_type": "proactive_interaction",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating proactive interaction: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "ProactiveInteractionGrader",
    "DEFAULT_PROACTIVE_INTERACTION_TEMPLATE",
]
