# -*- coding: utf-8 -*-
"""
Response Helpfulness Grader

Evaluates whether the agent's response is helpful, actionable, and provides value
beyond a minimal answer.
"""

import textwrap
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
RESPONSE_HELPFULNESS_PROMPT_EN = textwrap.dedent(
    """You are an expert in evaluating AI agent responses. Your task is to evaluate whether the agent's response is helpful and provides genuine value to the user. A helpful response goes beyond merely being correct — it is actionable, clear, and addresses the user's underlying needs.

<Rubrics>
1. The response directly addresses the user's underlying intent, not just the surface-level query
2. The response provides actionable information or concrete next steps the user can take
3. The response offers relevant explanations or context that helps the user understand the answer
4. The response anticipates and addresses potential follow-up questions or concerns
5. The response provides appropriate caveats, limitations, or alternative approaches when relevant
6. The response is clear and well-organized, making it easy for the user to extract the key information
</Rubrics>

<Steps>
1. Identify user intent: What does the user really need from this interaction?
2. Assess actionability: Can the user act on the information provided?
3. Check value-add: Does the response provide value beyond a minimal or generic answer?
4. Evaluate clarity: Is the response well-structured and easy to understand?
5. Consider proactiveness: Does the response anticipate related needs or potential issues?
</Steps>

<Scale>
- **Score 5**: Highly helpful — Provides actionable, clear, and comprehensive information that thoroughly addresses the user's needs, with relevant context and proactive suggestions
- **Score 4**: Helpful — Addresses the user's needs well with actionable information, but could offer slightly more context or proactiveness
- **Score 3**: Moderately helpful — Provides a correct but minimal answer without significant value-add, context, or proactive guidance
- **Score 2**: Slightly helpful — Addresses the query partially but lacks actionability, clarity, or important context
- **Score 1**: Not helpful — The response fails to provide useful or actionable information for the user
</Scale>

<User Query>
{query}
</User Query>

<Agent Response>
{response}
</Agent Response>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of response helpfulness, including strengths and areas for improvement>",
    "score": <integer between 1 and 5>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
RESPONSE_HELPFULNESS_PROMPT_ZH = textwrap.dedent(
    """你是一名评估AI智能体回复的专家。你的任务是评估智能体的回复是否有帮助且为用户提供了真正的价值。有帮助的回复不仅仅是正确——它还应该是可操作的、清晰的，并满足用户的潜在需求。

<评分标准>
1. 回复直接针对用户的潜在意图，而不仅仅是表面查询
2. 回复提供可操作的信息或用户可以采取的具体步骤
3. 回复提供相关的解释或背景，帮助用户理解答案
4. 回复预测并处理潜在的后续问题或关注点
5. 回复在相关时提供适当的注意事项、限制或替代方法
6. 回复清晰且组织良好，便于用户提取关键信息
</评分标准>

<评估步骤>
1. 识别用户意图：用户真正需要从这次交互中获得什么？
2. 评估可操作性：用户能否根据提供的信息采取行动？
3. 检查价值增值：回复是否提供了超越最小化或通用答案的价值？
4. 评估清晰度：回复是否结构良好且易于理解？
5. 考虑主动性：回复是否预测了相关需求或潜在问题？
</评估步骤>

<评分量表>
- **分数 5**：非常有帮助 — 提供了可操作、清晰且全面的信息，充分满足用户需求，包含相关背景和主动建议
- **分数 4**：有帮助 — 很好地满足了用户需求并提供可操作信息，但可以在背景或主动性方面更完善
- **分数 3**：有一定帮助 — 提供了正确但最小化的答案，没有显著的价值增值、背景或主动指导
- **分数 2**：略有帮助 — 部分处理了查询，但缺乏可操作性、清晰度或重要背景
- **分数 1**：没有帮助 — 回复未能为用户提供有用或可操作的信息
</评分量表>

<用户查询>
{query}
</用户查询>

<智能体回复>
{response}
</智能体回复>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于回复有用性的详细解释，包括优点和改进空间>",
    "score": <1 到 5 之间的整数>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_RESPONSE_HELPFULNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=RESPONSE_HELPFULNESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=RESPONSE_HELPFULNESS_PROMPT_ZH,
            ),
        ],
    },
)


class ResponseHelpfulnessGrader(LLMGrader):
    """
    Response Helpfulness Grader

    Evaluates whether the agent's response is helpful, actionable, and provides value
    beyond a minimal answer.

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
        >>> grader = ResponseHelpfulnessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="How do I deploy a Python app?",
        ...     response="You can deploy using Docker. Here are the steps: 1) Create a Dockerfile..."
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_RESPONSE_HELPFULNESS_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ResponseHelpfulnessGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="response_helpfulness",
            mode=GraderMode.POINTWISE,
            description="Evaluate response helpfulness and actionability",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        query: Union[str, List[Dict[str, Any]]],
        response: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate response helpfulness.

        Args:
            query: User query or chat history
            response: Agent's final response to evaluate
            context: Optional task context

        Returns:
            GraderScore: Score between 1 and 5
        """
        # Format query as string
        if isinstance(query, list):
            query_str = "\n".join(
                [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in query],
            )
        else:
            query_str = str(query)

        context_str = context if context else ""

        try:
            result = await super()._aevaluate(
                query=query_str,
                response=response,
                context=context_str,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating response helpfulness: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "response_helpfulness",
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ResponseHelpfulnessGrader",
    "DEFAULT_RESPONSE_HELPFULNESS_TEMPLATE",
]
