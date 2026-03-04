# -*- coding: utf-8 -*-
"""
Self Correction Grader for Multi-turn Conversations (SC)

Evaluates whether the assistant can appropriately correct its errors when
the user provides reasonable feedback or challenges incorrect information.

This is a reflection ability that tests the model's capacity to:
- Recognize when user feedback indicates an error
- Acknowledge mistakes gracefully
- Provide corrected information accurately
- Distinguish between valid corrections and invalid challenges
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

SELF_CORRECTION_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的自我修正能力评估专家。您的任务是评估多轮对话中，当用户对助手的回答提出合理质疑时，助手是否能够恰当地承认错误并提供正确的信息。

<评分标准>
完美的自我修正应该：
- 识别出自己之前的回答确实存在错误。
- 在承认错误时表现得谦逊、专业，不推卸责任。
- 提供准确、完整的修正信息。
- 清楚地解释错误所在以及正确答案是什么。
- 表现出从错误中学习的态度。
</评分标准>

<评估步骤>
1. 分析对话历史，确定助手之前是否确实给出了错误的信息。
2. 判断用户的质疑是否合理且指出了真正的错误。
3. 评估助手是否承认了错误（如果确实有错）。
4. 检查修正后的信息是否准确。
5. 评估态度是否恰当（不过度道歉也不狡辩）。
</评估步骤>

<注意事项>
如果助手原本就是对的，而用户的质疑是错误的，这属于"自我肯定"场景，不在此评估范围。本评估假设用户的质疑是合理的，助手确实需要修正。
</注意事项>

<评分量表>
- 5分：完美修正，助手恰当承认错误，提供了准确的修正信息，解释清晰，态度专业。
- 4分：良好修正，助手承认错误并修正，信息基本准确，但解释或态度略有不足。
- 3分：基本修正，助手进行了修正，但修正不够完整或态度不够恰当。
- 2分：修正不足，助手勉强承认错误，但修正信息仍有问题或态度不佳。
- 1分：拒绝修正，助手坚持错误观点，拒绝承认错误，或修正后仍然错误。
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
    "reason": "详细说明评估理由，包括错误识别、修正质量和态度分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


SELF_CORRECTION_PROMPT_EN = textwrap.dedent(
    """
You are a professional Self-Correction Evaluation Expert. Your task is to evaluate
whether the assistant can appropriately acknowledge errors and provide correct
information when users provide reasonable feedback or challenges.

<Rubrics>
Perfect self-correction should:
- Recognize that its previous response was indeed incorrect.
- Acknowledge errors humbly and professionally without deflecting.
- Provide accurate and complete corrected information.
- Clearly explain what was wrong and what the correct answer is.
- Demonstrate a learning attitude from the mistake.
</Rubrics>

<Steps>
1. Analyze conversation history to determine if the assistant actually provided incorrect information before.
2. Determine if the user's challenge is reasonable and pointing out a real error.
3. Evaluate if the assistant acknowledged the error (if there was one).
4. Check if the corrected information is accurate.
5. Evaluate if the attitude is appropriate (not over-apologizing or being defensive).
</Steps>

<Constraints>
If the assistant was originally correct and the user's challenge is wrong, this is a
"self-affirmation" scenario, not evaluated here. This evaluation assumes the user's
challenge is reasonable and the assistant needs to correct.
</Constraints>

<Scale>
- 5: Perfect correction - appropriately acknowledged error, provided accurate correction,
  clear explanation, professional attitude.
- 4: Good correction - acknowledged and corrected error, mostly accurate, but
  explanation or attitude slightly lacking.
- 3: Basic correction - made correction but incomplete or attitude not quite appropriate.
- 2: Insufficient correction - reluctantly acknowledged error, but correction still problematic or poor attitude.
- 1: Refused correction - insisted on wrong view, refused to acknowledge error, or still wrong after correction.
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
    "reason": "detailed explanation including error recognition, correction quality, and attitude analysis",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_SELF_CORRECTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=SELF_CORRECTION_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=SELF_CORRECTION_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class SelfCorrectionGrader(LLMGrader):
    """
    Self Correction Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can appropriately correct its errors when
        users provide reasonable feedback or challenges, demonstrating humility
        and commitment to accuracy.

    What it evaluates:
        - Error Recognition: Identifying when previous response was incorrect
        - Graceful Acknowledgment: Admitting mistakes professionally
        - Correction Accuracy: Providing correct information after error
        - Clear Explanation: Explaining what was wrong and why
        - Professional Attitude: Appropriate tone without over-apologizing

    When to use:
        - Evaluating error handling in chatbots
        - Testing self-correction capabilities
        - Assessing response to user feedback
        - Quality assurance for knowledge-based systems
        - Measuring conversational humility

    Scoring:
        - 5: Perfect correction - acknowledged, accurate, clear, professional
        - 4: Good correction - mostly accurate with minor gaps
        - 3: Basic correction - incomplete or attitude issues
        - 2: Insufficient correction - reluctant or still problematic
        - 1: Refused correction - insisted on error or defensive

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_SELF_CORRECTION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect correction, 1 = refused
            - reason: Explanation of correction quality analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import SelfCorrectionGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SelfCorrectionGrader(model=model)
        >>>
        >>> # Evaluate self-correction
        >>> history = [
        ...     {"role": "user", "content": "What's the capital of Australia?"},
        ...     {"role": "assistant", "content": "The capital of Australia is Sydney."},  # Wrong!
        ...     {"role": "user", "content": "I think that's incorrect."},
        ... ]
        >>> response = "You're right, I apologize. The capital of Australia is Canberra, not Sydney."
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: high score for good correction
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
        Initialize SelfCorrectionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_SELF_CORRECTION_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="self_correction",
            mode=GraderMode.POINTWISE,
            description="Evaluate self-correction ability in multi-turn conversations",
            template=template or DEFAULT_SELF_CORRECTION_TEMPLATE,
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
        Evaluate self-correction of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 self-correction score
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
                "evaluation_type": "self_correction",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating self-correction: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "SelfCorrectionGrader",
    "DEFAULT_SELF_CORRECTION_TEMPLATE",
]
