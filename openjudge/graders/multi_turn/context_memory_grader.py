# -*- coding: utf-8 -*-
"""
Context Memory Grader for Multi-turn Conversations (CM)

Evaluates whether the assistant can recall early conversation details and
maintain content coherence throughout the dialogue.

This is a fundamental perception ability that tests the model's capacity to:
- Remember facts, preferences, and constraints mentioned earlier
- Maintain consistency with previous statements
- Reference and build upon earlier conversation context
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

CONTEXT_MEMORY_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的上下文记忆能力评估专家。您的任务是评估多轮对话中助手是否能够准确回忆和利用早期对话中的细节信息，保持内容的连贯性。

<评分标准>
完美的上下文记忆应该：
- 准确记住用户在早期对话中提到的具体事实、数据或信息。
- 记住用户表达的偏好、喜好或限制条件。
- 恰当地引用或建立在之前对话内容的基础上。
- 在整个对话过程中保持信息的一致性，不出现遗忘或矛盾。
- 当需要引用早期信息时，能准确无误地复述关键细节。
</评分标准>

<评估步骤>
1. 仔细阅读对话历史，识别用户在早期轮次中提供的关键信息点。
2. 检查当前回复是否需要引用或依赖这些早期信息。
3. 评估助手是否准确记住了相关的早期信息。
4. 评估助手是否正确地将这些信息应用到当前回复中。
5. 检查是否存在遗忘或错误引用早期细节的情况。
</评估步骤>

<注意事项>
如果当前回复不需要引用早期信息，但回复本身是合理的，不应扣分。
</注意事项>

<评分量表>
- 5分：完美记忆，助手准确记住并恰当运用了所有相关的早期对话细节。
- 4分：良好记忆，助手记住了大部分关键信息，仅有极小的遗漏或不精确。
- 3分：基本记忆，助手记住了主要信息，但遗漏了一些重要细节。
- 2分：记忆不足，助手遗忘了多个重要的早期信息，影响了回复质量。
- 1分：严重遗忘，助手几乎完全忽略了早期对话内容，回复与上下文脱节。
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
    "reason": "详细说明评估理由，包括助手记住或遗忘了哪些关键信息",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


CONTEXT_MEMORY_PROMPT_EN = textwrap.dedent(
    """
You are a professional Context Memory Evaluation Expert. Your task is to evaluate
whether the assistant can accurately recall and utilize details from earlier parts of
a multi-turn conversation, maintaining content coherence.

<Rubrics>
Perfect context memory should:
- Accurately remember specific facts, data, or information mentioned by the user earlier.
- Remember user preferences, likes, or constraints expressed earlier.
- Appropriately reference or build upon earlier conversation content.
- Maintain information consistency throughout the conversation without forgetting or contradicting.
- Accurately restate key details when referencing earlier information.
</Rubrics>

<Steps>
1. Carefully read the conversation history and identify key information points provided by the user in earlier turns.
2. Check whether the current response needs to reference or rely on this earlier information.
3. Evaluate whether the assistant accurately remembered relevant earlier information.
4. Evaluate whether the assistant correctly applied this information in the current response.
5. Check for any forgotten or incorrectly referenced earlier details.
</Steps>

<Constraints>
If the current response doesn't need to reference earlier information but is otherwise reasonable, do not penalize.
</Constraints>

<Scale>
- 5: Perfect memory - accurately remembered and appropriately used all relevant earlier conversation details.
- 4: Good memory - remembered most key information with only minor omissions or imprecisions.
- 3: Basic memory - remembered main information but missed some important details.
- 2: Insufficient memory - forgot multiple important earlier details, affecting response quality.
- 1: Severe forgetting - almost completely ignored earlier conversation content, response disconnected from context.
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
    "reason": "detailed explanation including what key information the assistant remembered or forgot",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_CONTEXT_MEMORY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=CONTEXT_MEMORY_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=CONTEXT_MEMORY_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class ContextMemoryGrader(LLMGrader):
    """
    Context Memory Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can recall early conversation details and
        maintain content coherence throughout the dialogue, ensuring consistent
        and contextually aware responses.

    What it evaluates:
        - Fact Retention: Remembering specific facts and data from earlier turns
        - Preference Memory: Recalling user preferences and constraints
        - Contextual Building: Appropriately referencing earlier content
        - Consistency: Maintaining information consistency without contradictions
        - Detail Accuracy: Correctly restating key details when needed

    When to use:
        - Evaluating long-context conversation handling
        - Testing memory capabilities in dialogue systems
        - Assessing context retention in chatbots
        - Quality assurance for multi-turn interactions
        - Identifying context loss patterns

    Scoring:
        - 5: Perfect memory - all relevant details accurately remembered
        - 4: Good memory - most key information retained
        - 3: Basic memory - main information remembered with some gaps
        - 2: Insufficient memory - multiple important details forgotten
        - 1: Severe forgetting - response disconnected from context

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_CONTEXT_MEMORY_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect memory, 1 = severe forgetting
            - reason: Explanation of memory analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import ContextMemoryGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = ContextMemoryGrader(model=model)
        >>>
        >>> # Evaluate context memory
        >>> history = [
        ...     {"role": "user", "content": "I'm allergic to nuts and prefer vegetarian food."},
        ...     {"role": "assistant", "content": "I'll keep that in mind for recommendations."},
        ...     {"role": "user", "content": "What should I order at this restaurant?"},
        ... ]
        >>> response = "I recommend the grilled salmon with almond sauce."  # Forgot nut allergy!
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: low score due to forgetting constraint
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
        Initialize ContextMemoryGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_CONTEXT_MEMORY_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="context_memory",
            mode=GraderMode.POINTWISE,
            description="Evaluate context memory ability in multi-turn conversations",
            template=template or DEFAULT_CONTEXT_MEMORY_TEMPLATE,
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
        Evaluate context memory of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 context memory score
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
                "evaluation_type": "context_memory",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating context memory: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "ContextMemoryGrader",
    "DEFAULT_CONTEXT_MEMORY_TEMPLATE",
]
