# -*- coding: utf-8 -*-
"""
Topic Switch Grader for Multi-turn Conversations (TS)

Evaluates whether the assistant can recognize when the user suddenly switches
to a new topic and appropriately focus on the new topic while maintaining
the ability to return to previous topics when needed.

This is a context interference resistance ability that tests the model's capacity to:
- Detect sudden topic changes in conversation
- Appropriately shift focus to new topics
- Not confuse information from different topics
- Handle topic switches gracefully
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

TOPIC_SWITCH_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的话题切换能力评估专家。您的任务是评估多轮对话中助手是否能够识别用户突然切换话题的情况，并恰当地聚焦于新话题进行回应。

<评分标准>
完美的话题切换处理应该：
- 准确识别用户已经切换到了一个新的话题。
- 将注意力恰当地转移到新话题上。
- 避免将之前话题的信息错误地混入新话题的回复中。
- 对新话题的回复准确、相关且有帮助。
- 在处理话题切换时表现得自然流畅。
</评分标准>

<评估步骤>
1. 分析对话历史，识别用户是否突然提出了与之前完全不同的问题或话题。
2. 判断新话题与旧话题之间是否有明显的断裂。
3. 评估助手是否正确识别了话题切换。
4. 检查助手是否针对新话题给出了恰当的回应。
5. 检查是否错误地将旧话题的信息带入新话题。
</评估步骤>

<注意事项>
如果用户后来又回到之前的话题，助手应能正确处理。助手不应在不必要的情况下主动提及旧话题。
</注意事项>

<评分量表>
- 5分：完美切换，助手准确识别话题变化，回复完全聚焦于新话题，信息准确相关。
- 4分：良好切换，助手正确识别并回应新话题，仅有轻微的旧话题残留。
- 3分：基本切换，助手大体上回应了新话题，但存在一些混淆或不够聚焦。
- 2分：切换不足，助手未能很好地识别话题变化，回复中混入了较多旧话题内容。
- 1分：切换失败，助手完全忽略话题切换，继续回应旧话题或严重混淆两个话题。
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
    "reason": "详细说明评估理由，包括话题切换的识别和处理分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


TOPIC_SWITCH_PROMPT_EN = textwrap.dedent(
    """
You are a professional Topic Switch Evaluation Expert. Your task is to evaluate
whether the assistant can recognize when the user suddenly switches to a new topic and
appropriately focus on the new topic.

<Rubrics>
Perfect topic switch handling should:
- Accurately identify that the user has switched to a new topic.
- Appropriately shift attention to the new topic.
- Avoid incorrectly mixing information from previous topics into the new topic response.
- Provide accurate, relevant, and helpful response to the new topic.
- Handle topic switches naturally and smoothly.
</Rubrics>

<Steps>
1. Analyze conversation history to identify if the user suddenly raised a completely different question or topic.
2. Determine if there is a clear break between the new and old topics.
3. Evaluate if the assistant correctly identified the topic switch.
4. Check if the assistant provided an appropriate response to the new topic.
5. Check if old topic information was incorrectly brought into the new topic.
</Steps>

<Constraints>
If the user later returns to a previous topic, the assistant should handle it correctly.
The assistant should not unnecessarily mention old topics.
</Constraints>

<Scale>
- 5: Perfect switch - accurately identified topic change, response fully focused on new topic, accurate and relevant.
- 4: Good switch - correctly identified and responded to new topic with only slight remnants of old topic.
- 3: Basic switch - generally responded to new topic but with some confusion or lack of focus.
- 2: Insufficient switch - failed to properly identify topic change, response mixed with old topic content.
- 1: Failed switch - completely ignored topic switch, continued responding to old topic or severely confused both.
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
    "reason": "detailed explanation including analysis of topic switch recognition and handling",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_TOPIC_SWITCH_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=TOPIC_SWITCH_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=TOPIC_SWITCH_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class TopicSwitchGrader(LLMGrader):
    """
    Topic Switch Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can recognize sudden topic changes and
        appropriately focus on new topics while maintaining context awareness,
        avoiding confusion between different conversation threads.

    What it evaluates:
        - Topic Detection: Recognizing when user switches to a new topic
        - Focus Shift: Appropriately redirecting attention to new topic
        - Information Isolation: Not mixing old topic info into new responses
        - Response Relevance: Providing accurate answers to new topic
        - Natural Handling: Smooth and graceful topic transitions

    When to use:
        - Evaluating context management in chatbots
        - Testing topic tracking capabilities
        - Assessing conversation flexibility
        - Quality assurance for multi-topic dialogues
        - Measuring context interference resistance

    Scoring:
        - 5: Perfect switch - accurate detection, focused response
        - 4: Good switch - correct handling with minor remnants
        - 3: Basic switch - some confusion or lack of focus
        - 2: Insufficient switch - mixed old topic content
        - 1: Failed switch - ignored topic change or severe confusion

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_TOPIC_SWITCH_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect switch, 1 = failed
            - reason: Explanation of topic switch handling analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import TopicSwitchGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = TopicSwitchGrader(model=model)
        >>>
        >>> # Evaluate topic switch handling
        >>> history = [
        ...     {"role": "user", "content": "What's the best programming language?"},
        ...     {"role": "assistant", "content": "It depends on your use case. Python is great for ML."},
        ...     {"role": "user", "content": "By the way, what's the weather like in Tokyo?"},  # Topic switch!
        ... ]
        >>> response = "I don't have real-time weather data, but Tokyo typically has mild weather in spring."
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: high score for handling topic switch
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
        Initialize TopicSwitchGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_TOPIC_SWITCH_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="topic_switch",
            mode=GraderMode.POINTWISE,
            description="Evaluate topic switch handling ability in multi-turn conversations",
            template=template or DEFAULT_TOPIC_SWITCH_TEMPLATE,
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
        Evaluate topic switch handling of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 topic switch handling score
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
                "evaluation_type": "topic_switch",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating topic switch: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "TopicSwitchGrader",
    "DEFAULT_TOPIC_SWITCH_TEMPLATE",
]
