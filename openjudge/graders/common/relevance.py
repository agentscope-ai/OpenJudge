# -*- coding: utf-8 -*-
"""
Relevance Grader

Evaluates how relevant a response is to the user's query in the conversation history.
"""

import textwrap
from typing import Optional

from loguru import logger

from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
RELEVANCE_PROMPT_EN = textwrap.dedent(
    """
You are a professional data annotator responsible for evaluating how relevant the model response is to the user's query. Your task is to score according to the following criteria:

<Scoring Criteria>
A highly relevant response should:
- Directly address the user's question or request.
- Provide information that is on-topic and pertinent to the query.
- Include sufficient detail to satisfy the user's information needs.
- Stay focused without drifting to unrelated topics.
- For multi-turn conversations, maintain context awareness from previous exchanges.

Points should be deducted for:
- Completely off-topic or unrelated responses.
- Vague or superficial answers that lack specific information.
- Partial responses that omit key information requested.
- Responses that acknowledge the query but fail to provide useful content.
- Generic statements that don't specifically address the question.
</Scoring Criteria>

<Guidance>
- Carefully read the query (or conversation history) and model response.
- Determine if the response directly addresses what the user is asking.
- Check if the information provided is complete, partial, or missing.
- Assess whether the response stays on-topic or includes irrelevant content.
- For conversations, consider whether the response maintains context from earlier turns.
- The score should reflect how well the response aligns with the user's information needs.
</Guidance>

<Reminder>
The goal is to evaluate relevance to the query, not overall quality.
A score of 5 means the response is highly relevant and comprehensive.
A score of 1 means the response is completely irrelevant to the query.
</Reminder>
<query>
{query}
</query>

<response>
{response}
</response>

Additional context (ignore if empty):
<context>
{context}
</context>

The following is the correct response for your reference (ignore if empty):
<reference_response>
{reference_response}
</reference_response>

# Output Instructions
**Note**: If a reference response is provided, you may use it as a baseline for comparison to better assess the quality and relevance of the evaluated response.

Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5, where 5 means highly relevant and 1 means completely irrelevant>,
    "reason": "<brief explanation for the assigned score, specifically mentioning how the response addresses or fails to address the query>"
}}

Scoring Scale:
- 5: Perfectly relevant: the response completely fulfills the user's search intent, accurately answering the question or providing the required information.
- 4: Highly relevant: the response largely meets the search requirements, possibly lacking some details or having minor inaccuracies, but still a high-quality and directly relevant result.
- 3: Partially relevant: the response has some connection to the query but does not fully meet the requirements; the user may need to further filter or supplement the information.
- 2: Weakly relevant: the response has only a weak connection to the query, possibly covering the same topic but deviating from the core intent, and has low practical value.
- 1: Irrelevant: the response is completely unrelated to the query, or contains misleading or incorrect information.

JSON:
"""
).strip()

# Chinese Prompt
RELEVANCE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的数据标注员，负责评估模型输出与用户查询的相关性。你的任务是根据以下标准进行评分：

<评分标准>
高度相关的回答应该：
- 直接解决用户的问题或请求。
- 提供与查询主题相关且切题的信息。
- 包含足够的细节以满足用户的信息需求。
- 保持专注，不偏离到无关主题。
- 对于多轮对话，保持对先前交流的上下文意识。

以下情况应扣分：
- 完全偏离主题或无关的回答。
- 模糊或肤浅的答案，缺乏具体信息。
- 部分回答，遗漏了请求的关键信息。
- 承认查询但未能提供有用内容的回答。
- 通用陈述，没有具体解决问题。
</评分标准>

<指导>
- 仔细阅读查询（或对话历史）和模型输出。
- 判断输出是否直接解决了用户所询问的内容。
- 检查提供的信息是完整的、部分的还是缺失的。
- 评估输出是否保持主题或包含无关内容。
- 对于对话，考虑输出是否保持了早期轮次的上下文。
- 分数应反映输出与用户信息需求的契合程度。
</指导>

<提醒>
目标是评估与查询的相关性，而不是整体质量。
分数5表示回答高度相关且全面。
分数1表示回答与查询完全无关。
</提醒>

<查询>
{query}
</查询>

<回答>
{response}
</回答>

附加上下文（如为空则忽略）:
<上下文>
{context}
</上下文>

参考回答（用于比较，如为空则忽略）：
<参考回答>
{reference_response}
</参考回答>

# 输出指令
**注意**：如果提供了参考回答，你可以将其作为基准进行比较，以更好地评估被评价回答的质量和相关性。

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1到5之间的整数，其中5表示高度相关，1表示完全不相关>,
    "reason": "<对所给分数的简要解释，特别提到输出如何解决或未能解决查询>"
}}

评分标尺：
- 5: 完全相关，回答完全满足用户查询意图，精准回答问题或提供所需信息。
- 4: 高度相关，回答基本满足查询需求，可能略缺细节或略有偏差，但仍是高质量、直接相关的结果。
- 3: 部分相关，回答与查询有一定关联，但未完全满足需求，可能需要用户进一步筛选或补充信息。
- 2: 弱相关，回答与查询仅有微弱联系，可能涉及相同主题但偏离核心意图，实用价值较低。
- 1: 不相关，回答与查询完全无关，或存在误导、错误匹配。

JSON:
"""
).strip()


# Build default template from prompts
DEFAULT_RELEVANCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=RELEVANCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=RELEVANCE_PROMPT_ZH,
            ),
        ],
    },
)


class RelevanceGrader(LLMGrader):
    """
    Relevance Grader

    Purpose:
        Evaluates how relevant and appropriate a response is to the user's query within
        the conversation history. Assesses whether the response directly addresses the
        question, provides sufficient information, and stays on-topic.

    What it evaluates:
        - Direct relevance: Does the response address the user's query?
        - Completeness: Is the information complete or partial?
        - Specificity: Is it vague/generic or specific/insightful?
        - On-topic: Does it stay focused on the query or drift off-topic?
        - Context awareness: For multi-turn conversations, does it consider conversation history?

    When to use:
        - Evaluating chatbot and assistant response relevance
        - Filtering out off-topic or unhelpful responses
        - Quality assurance for conversational AI systems
        - Training reward models to prefer relevant responses
        - A/B testing response generation strategies for relevance

    Scoring:
        - 5: Perfectly relevant response with insights that enhance understanding
        - 4: Highly relevant and sufficient response covering all essential aspects
        - 3: Partially relevant response, missing some key details
        - 2: Weakly relevant response, lacks meaningful information
        - 1: Irrelevant response, off-topic or unrelated to the query

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_RELEVANCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = highly relevant, 1 = irrelevant
            - reason: Explanation of relevance assessment
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.common.relevance import RelevanceGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = RelevanceGrader(model=model, threshold=0.7)
        >>>
        >>> # Relevant response
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="What are Python decorators?",
        ...     response="Decorators are functions that modify other functions. They use @syntax..."
        ... ))
        >>> print(result.score)  # 5 - directly answers the question with details
        >>>
        >>> # Irrelevant response
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="What are Python decorators?",
        ...     response="I like programming in various languages.",
        ... ))
        >>> print(result.score)  # 1 - completely off-topic
        >>>
        >>> # With context
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="What's the weather like then?",
        ...     response="July is summer in Europe with warm weather...",
        ...     context="Previous conversation about planning a July vacation to Europe"
        ... ))
        >>> print(result.score)  # 5 - relevant with conversation context
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_RELEVANCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize RelevanceGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_RELEVANCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="relevance",
            mode=GraderMode.POINTWISE,
            description="Evaluate relevance of response to user query",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        reference_response: str = "",
    ) -> GraderScore:
        """
        Evaluate relevance of response to query

        Args:
            query: Input query or conversation history
            response: Model response to evaluate
            context: Additional context or background information. Defaults to empty string.
            reference: Reference response for comparison. Defaults to empty string.

        Returns:
            GraderScore: Score with relevance value [1, 5]
                        where 5 means highly relevant, 1 means irrelevant

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What is machine learning?",
            ...     response="Machine learning is a subset of AI that enables systems to learn from data...",
            ...     context="User is a beginner asking for a simple explanation",
            ... )
        """
        try:
            result = await super().aevaluate(
                query=query,
                response=response,
                context=context,
                reference_response=reference_response,
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={"threshold": self.threshold},
            )

        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["RelevanceGrader", "DEFAULT_RELEVANCE_TEMPLATE"]
