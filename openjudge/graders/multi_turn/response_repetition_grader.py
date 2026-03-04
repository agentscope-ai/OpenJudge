# -*- coding: utf-8 -*-
"""
Response Repetition Grader for Multi-turn Conversations (RR)

Evaluates whether the current assistant response contains repetitive content
compared to earlier responses in the conversation history.

This grader detects:
- Verbatim repetition of previous responses
- Paraphrased repetition with similar meaning
- Redundant information that was already provided
- Circular responses that don't add new value
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

RESPONSE_REPETITION_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的回复重复检测评估专家。您的任务是评估多轮对话中助手当前的回复是否与历史对话中的回复存在重复或冗余。

<评分标准>
无重复的高质量回复应该：
- 提供新的、有价值的信息，而不是重复之前已经说过的内容。
- 即使涉及相同话题，也能从新的角度或深度进行阐述。
- 避免逐字重复或高度相似的表述。
- 在需要重申某些信息时，能够以更简洁或更有针对性的方式表达。
- 推进对话发展，而不是原地踏步。
</评分标准>

<评估步骤>
1. 仔细阅读对话历史中助手的所有回复。
2. 将当前回复与历史回复进行对比。
3. 检查是否存在逐字重复的内容。
4. 检查是否存在意思相同但表述不同的重复（改写重复）。
5. 评估重复内容的比例和对回复质量的影响。
</评估步骤>

<注意事项>
- 如果用户重复问了相同的问题，助手重复回答是合理的，不应扣分。
- 必要的信息确认或强调不算重复。
- 重点关注不必要的、冗余的重复内容。
- 考虑重复是否影响了用户体验和对话效率。
</注意事项>

<评分量表>
- 5分：无重复，回复完全是新的、有价值的内容，有效推进了对话。
- 4分：极少重复，可能有极小部分内容与之前相似，但整体提供了新信息。
- 3分：部分重复，回复中有一些内容与之前重复，但仍包含新的有用信息。
- 2分：明显重复，回复中大部分内容是之前已经说过的，新信息很少。
- 1分：严重重复，回复几乎完全是之前内容的重复，没有提供任何新价值。
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
    "reason": "详细说明评估理由，包括发现的重复内容及其影响分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


RESPONSE_REPETITION_PROMPT_EN = textwrap.dedent(
    """
You are a professional Response Repetition Detection Expert. Your task is to evaluate
whether the assistant's current response contains repetitive or redundant content
compared to earlier responses in the conversation history.

<Rubrics>
High-quality non-repetitive responses should:
- Provide new, valuable information rather than repeating what was already said.
- Even when addressing the same topic, offer new perspectives or depth.
- Avoid verbatim repetition or highly similar expressions.
- When reiterating information is necessary, express it more concisely or targeted.
- Advance the conversation rather than staying in place.
</Rubrics>

<Steps>
1. Carefully read all assistant responses in the conversation history.
2. Compare the current response with historical responses.
3. Check for verbatim repetition.
4. Check for semantic repetition (same meaning, different wording).
5. Evaluate the proportion of repeated content and its impact on response quality.
</Steps>

<Constraints>
- If the user asked the same question again, the assistant repeating the answer is
  reasonable and should not be penalized.
- Necessary information confirmation or emphasis does not count as repetition.
- Focus on unnecessary, redundant repetitive content.
- Consider whether repetition affects user experience and conversation efficiency.
</Constraints>

<Scale>
- 5: No repetition - response is entirely new and valuable content, effectively advancing the conversation.
- 4: Minimal repetition - may have very small portions similar to before, but overall provides new information.
- 3: Partial repetition - some content repeats earlier responses, but still contains new useful information.
- 2: Significant repetition - most content was already said before, very little new information.
- 1: Severe repetition - response is almost entirely repetition of previous content, provides no new value.
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
    "reason": "detailed explanation including identified repetitive content and impact analysis",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_RESPONSE_REPETITION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=RESPONSE_REPETITION_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=RESPONSE_REPETITION_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class ResponseRepetitionGrader(LLMGrader):
    """
    Response Repetition Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the current assistant response contains repetitive content
        compared to earlier responses in the conversation history, ensuring responses
        provide new value and advance the conversation.

    What it evaluates:
        - Verbatim Repetition: Exact or near-exact repetition of previous content
        - Semantic Repetition: Same meaning expressed with different wording
        - Redundant Information: Information already provided earlier
        - Conversation Progress: Whether the response advances the dialogue
        - Value Addition: New insights or information provided

    When to use:
        - Evaluating response quality in multi-turn conversations
        - Detecting circular or stagnant dialogue patterns
        - Assessing conversation efficiency
        - Quality assurance for chatbot responses
        - Identifying redundant content in dialogue systems

    Scoring:
        - 5: No repetition - entirely new and valuable content
        - 4: Minimal repetition - mostly new information
        - 3: Partial repetition - some new useful information
        - 2: Significant repetition - very little new information
        - 1: Severe repetition - no new value provided

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_RESPONSE_REPETITION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = no repetition, 1 = severe repetition
            - reason: Explanation of repetition analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import ResponseRepetitionGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = ResponseRepetitionGrader(model=model)
        >>>
        >>> # Evaluate response with repetition
        >>> history = [
        ...     {"role": "user", "content": "What is Python?"},
        ...     {"role": "assistant", "content": "Python is a high-level programming language."},
        ...     {"role": "user", "content": "Tell me more."},
        ... ]
        >>> response = "Python is a high-level programming language."  # Exact repeat!
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: low score due to repetition
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
        Initialize ResponseRepetitionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_RESPONSE_REPETITION_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="response_repetition",
            mode=GraderMode.POINTWISE,
            description="Evaluate response repetition in multi-turn conversations",
            template=template or DEFAULT_RESPONSE_REPETITION_TEMPLATE,
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
        Evaluate response repetition of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 repetition score (5 = no repetition, 1 = severe repetition)
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
                "evaluation_type": "response_repetition",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating response repetition: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "ResponseRepetitionGrader",
    "DEFAULT_RESPONSE_REPETITION_TEMPLATE",
]
