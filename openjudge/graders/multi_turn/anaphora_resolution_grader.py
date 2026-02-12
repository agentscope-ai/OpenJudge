# -*- coding: utf-8 -*-
"""
Anaphora Resolution Grader for Multi-turn Conversations (AR)

Evaluates whether the assistant can accurately identify and resolve pronouns
and demonstratives like "this", "it", "that", "they" to their correct referents
in the conversation context.

This is a fundamental perception ability that tests the model's capacity to:
- Correctly identify what pronouns refer to
- Maintain referential coherence across turns
- Handle complex anaphoric chains
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

ANAPHORA_RESOLUTION_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的指代消解能力评估专家。您的任务是评估多轮对话中助手是否能够准确识别和解析代词（如"这"、"它"、"那个"、"他们"等）所指代的具体对象。

<评分标准>
完美的指代消解应该：
- 正确识别用户话语中的代词和指示词。
- 准确判断这些代词所指代的具体实体或概念。
- 根据对话上下文正确推断指代关系。
- 针对正确的指代对象进行回应。
- 当存在多个可能的指代对象时，能做出合理的判断。
</评分标准>

<评估步骤>
1. 识别对话中出现的代词和指示词（如"这个"、"它"、"那"、"他们"、"前者"、"后者"等）。
2. 确定这些代词在上下文中应该指代的正确对象。
3. 检查助手是否正确理解了用户所指的对象。
4. 评估回复内容是否与正确的指代对象相关。
5. 检查是否存在指代错误导致的答非所问。
</评估步骤>

<注意事项>
特别关注跨轮次的指代关系、省略主语的情况、以及多个候选指代对象的情况。
</注意事项>

<评分量表>
- 5分：完美解析，助手准确理解了所有代词的指代对象，回复完全正确。
- 4分：良好解析，助手正确理解了主要的指代关系，仅有细微偏差。
- 3分：基本解析，助手理解了部分指代关系，但存在一些误解。
- 2分：解析不足，助手对关键代词的理解有误，导致回复偏离主题。
- 1分：严重错误，助手完全误解了指代对象，回复与用户意图无关。
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
    "reason": "详细说明评估理由，包括代词解析的正确性分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


ANAPHORA_RESOLUTION_PROMPT_EN = textwrap.dedent(
    """
You are a professional Anaphora Resolution Evaluation Expert. Your task is to evaluate
whether the assistant can accurately identify and resolve pronouns (such as "this", "it",
"that", "they") to their correct referents in the conversation context.

<Rubrics>
Perfect anaphora resolution should:
- Correctly identify pronouns and demonstratives in user utterances.
- Accurately determine what entities or concepts these pronouns refer to.
- Correctly infer referential relationships based on conversation context.
- Address the correct referent in the response.
- Make reasonable judgments when multiple possible referents exist.
</Rubrics>

<Steps>
1. Identify pronouns and demonstratives in the conversation (e.g., "this", "it", "that",
   "they", "the former", "the latter").
2. Determine what these pronouns should correctly refer to in context.
3. Check if the assistant correctly understood what the user was referring to.
4. Evaluate if the response content is relevant to the correct referent.
5. Check for reference errors causing irrelevant responses.
</Steps>

<Constraints>
Pay special attention to cross-turn referential relationships, cases with omitted
subjects, and situations with multiple candidate referents.
</Constraints>

<Scale>
- 5: Perfect resolution - accurately understood all pronoun referents, completely correct response.
- 4: Good resolution - correctly understood main referential relationships with only minor deviations.
- 3: Basic resolution - understood some referential relationships but with some misunderstandings.
- 2: Insufficient resolution - misunderstood key pronouns, causing response to deviate from topic.
- 1: Severe errors - completely misunderstood referents, response unrelated to user intent.
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
    "reason": "detailed explanation including analysis of pronoun resolution correctness",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_ANAPHORA_RESOLUTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=ANAPHORA_RESOLUTION_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=ANAPHORA_RESOLUTION_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class AnaphoraResolutionGrader(LLMGrader):
    """
    Anaphora Resolution Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can accurately identify and resolve pronouns
        and demonstratives to their correct referents in the conversation context,
        ensuring proper understanding of user intent.

    What it evaluates:
        - Pronoun Resolution: Correctly identifying what "it", "this", "that" refer to
        - Cross-turn References: Maintaining referential coherence across conversation turns
        - Demonstrative Resolution: Understanding "the former", "the latter", etc.
        - Omitted Subject Handling: Inferring subjects when not explicitly stated
        - Ambiguity Resolution: Making reasonable judgments with multiple candidates

    When to use:
        - Evaluating comprehension in multi-turn conversations
        - Testing referential understanding in dialogue systems
        - Assessing context tracking capabilities
        - Quality assurance for conversational AI
        - Identifying misunderstanding patterns in chatbots

    Scoring:
        - 5: Perfect resolution - all referents correctly understood
        - 4: Good resolution - main references correct with minor deviations
        - 3: Basic resolution - some misunderstandings present
        - 2: Insufficient resolution - key pronouns misunderstood
        - 1: Severe errors - completely wrong referent identification

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_ANAPHORA_RESOLUTION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect resolution, 1 = severe errors
            - reason: Explanation of pronoun resolution analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import AnaphoraResolutionGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = AnaphoraResolutionGrader(model=model)
        >>>
        >>> # Evaluate anaphora resolution
        >>> history = [
        ...     {"role": "user", "content": "I bought a laptop and a phone."},
        ...     {"role": "assistant", "content": "Great purchases!"},
        ...     {"role": "user", "content": "Is it good for programming?"},  # "it" = laptop
        ... ]
        >>> response = "Yes, the laptop is excellent for programming."
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: high score for correct resolution
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
        Initialize AnaphoraResolutionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_ANAPHORA_RESOLUTION_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="anaphora_resolution",
            mode=GraderMode.POINTWISE,
            description="Evaluate anaphora resolution ability in multi-turn conversations",
            template=template or DEFAULT_ANAPHORA_RESOLUTION_TEMPLATE,
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
        Evaluate anaphora resolution of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 anaphora resolution score
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
                "evaluation_type": "anaphora_resolution",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating anaphora resolution: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "AnaphoraResolutionGrader",
    "DEFAULT_ANAPHORA_RESOLUTION_TEMPLATE",
]
