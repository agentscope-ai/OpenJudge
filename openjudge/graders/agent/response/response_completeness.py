# -*- coding: utf-8 -*-
"""
Response Completeness Grader

Evaluates whether the agent's final response completely addresses all aspects
of the user's query, covering all sub-questions and constraints.
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
RESPONSE_COMPLETENESS_PROMPT_EN = textwrap.dedent(
    """You are an expert in evaluating AI agent responses. Your task is to evaluate whether the agent's response completely addresses all aspects of the user's query. A complete response covers all sub-questions, satisfies all constraints, and does not leave any part of the query unanswered.

<Rubrics>
1. The response addresses the main intent of the user's query
2. The response covers all sub-questions or sub-topics mentioned in the query
3. The response satisfies all explicit constraints or conditions stated in the query
4. The response does not omit any critical information requested by the user
5. The response provides sufficient detail for each aspect of the query (not just brief mentions)
</Rubrics>

<Steps>
1. Parse the query: Identify all sub-questions, constraints, and information requests
2. Map response coverage: Check which aspects are addressed and which are missing
3. Assess depth: For each addressed aspect, evaluate if the detail level is sufficient
4. Check constraints: Verify that all explicit constraints are satisfied
5. Identify gaps: List any aspects of the query that are unanswered or insufficiently addressed
</Steps>

<Scale>
- **Score 5**: Complete — All aspects of the query are fully addressed with sufficient detail
- **Score 4**: Mostly complete — All major aspects addressed, but minor gaps in detail or coverage
- **Score 3**: Partially complete — Some aspects addressed, but notable gaps exist
- **Score 2**: Incomplete — Only a few aspects addressed, major parts of the query are unanswered
- **Score 1**: Severely incomplete — The response fails to address the query meaningfully
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
    "reason": "<detailed explanation of response completeness, listing covered and missing aspects>",
    "score": <integer between 1 and 5>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
RESPONSE_COMPLETENESS_PROMPT_ZH = textwrap.dedent(
    """你是一名评估AI智能体回复的专家。你的任务是评估智能体的回复是否完整地解决了用户查询的所有方面。完整的回复应该涵盖所有子问题、满足所有约束条件，不遗漏查询的任何部分。

<评分标准>
1. 回复针对了用户查询的主要意图
2. 回复涵盖了查询中提到的所有子问题或子主题
3. 回复满足了查询中陈述的所有明确约束或条件
4. 回复没有遗漏用户请求的任何关键信息
5. 回复为查询的每个方面提供了充分的细节（不仅仅是简要提及）
</评分标准>

<评估步骤>
1. 解析查询：识别所有子问题、约束和信息请求
2. 映射回复覆盖：检查哪些方面被处理，哪些被遗漏
3. 评估深度：对于每个已处理的方面，评估细节水平是否足够
4. 检查约束：验证是否满足了所有明确约束
5. 识别缺口：列出查询中未回答或回答不充分的方面
</评估步骤>

<评分量表>
- **分数 5**：完整 — 查询的所有方面都得到了充分详细的处理
- **分数 4**：基本完整 — 所有主要方面都已处理，但在细节或覆盖范围上有小缺口
- **分数 3**：部分完整 — 处理了一些方面，但存在明显的缺口
- **分数 2**：不完整 — 仅处理了少数方面，查询的大部分内容未回答
- **分数 1**：严重不完整 — 回复未能有意义地解决查询
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
    "reason": "<关于回复完整性的详细解释，列出已覆盖和缺失的方面>",
    "score": <1 到 5 之间的整数>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_RESPONSE_COMPLETENESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=RESPONSE_COMPLETENESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=RESPONSE_COMPLETENESS_PROMPT_ZH,
            ),
        ],
    },
)


class ResponseCompletenessGrader(LLMGrader):
    """
    Response Completeness Grader

    Evaluates whether the agent's final response completely addresses all aspects
    of the user's query.

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
        >>> grader = ResponseCompletenessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="What's the weather in NYC and should I bring an umbrella?",
        ...     response="It's sunny in NYC, 72°F. No umbrella needed."
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_RESPONSE_COMPLETENESS_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ResponseCompletenessGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="response_completeness",
            mode=GraderMode.POINTWISE,
            description="Evaluate response completeness in addressing the query",
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
        Evaluate response completeness.

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
            logger.error(f"Error evaluating response completeness: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "response_completeness",
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ResponseCompletenessGrader",
    "DEFAULT_RESPONSE_COMPLETENESS_TEMPLATE",
]
