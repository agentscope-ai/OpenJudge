# -*- coding: utf-8 -*-
"""
Search Timeliness Grader for Finance Domain

Evaluates the timeliness of stock search results by comparing two responses
based on currency of information and temporal notation.
"""

import textwrap
from typing import Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderRank
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# English Prompt
SEARCH_TIMELINESS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Search Timeliness Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Search Timeliness**:
Criterion 1. Search Timeliness: Assess the timeliness of stocks and information retrieved in the answer, including but not limited to:
a. Company data, capacity layout, policies, and industry events used should be the latest publicly available information, especially content with high real-time requirements such as macroeconomic policy changes and international trade agreement implementations.
b. For facts with obvious timeliness (such as tariff agreement effective dates, recent capacity commissioning), time points should be annotated to allow users to judge reference value.
</Rubrics>

<Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Steps>

<Constraints>
- **Objectivity**: Do not be influenced by the politeness or length of the response.
- **Focus**: Only evaluate based on the provided evaluation criteria.
</Constraints>

<Scale>
- If Answer 1 is better: rank = [1, 2]
- If Answer 2 is better: rank = [2, 1]
</Scale>

<Query>
{query}
</Query>

<Response 1>
{answer_1}
</Response 1>

<Response 2>
{answer_2}
</Response 2>

<Output Schema>
Please output your assessment in the following JSON format strictly:
{{
    "reason": "<detailed explanation of your evaluation reasoning, including performance comparison under each evaluation criterion>",
    "rank": <[1, 2] or [2, 1]>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SEARCH_TIMELINESS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**检索时效性质量**。

<评分标准>
请根据**检索时效性**评估回答：
评价标准1. 检索时效性：评估答案中检索到的股票和问题的时效性，包括但不限于：
a. 采用的公司数据、产能布局、政策和行业事件需为最新公开信息，特别是宏观政策变化和国际贸易协议落地等实时性要求高的内容。
b. 对有明显时效性的事实（如关税协议生效日期、近期产能投产）需标注时间节点，以便用户判断参考价值。
</评分标准>

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由;
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<注意事项>
- **客观性**：不要被回答的礼貌程度或长度所影响。
- **聚焦**：仅根据提供的评估标准进行评估。
</注意事项>

<评分量表>
- 如果回答1更好：rank = [1, 2]
- 如果回答2更好：rank = [2, 1]
</评分量表>

<查询>
{query}
</查询>

<回复1>
{answer_1}
</回复1>

<回复2>
{answer_2}
</回复2>

<输出格式>
请按以下结构化 JSON 格式严格输出你的评估：
{{
    "reason": "<详细解释你的评估理由，包括在各个评估标准下的表现对比>",
    "rank": <[1, 2] 或 [2, 1]>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SEARCH_TIMELINESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SEARCH_TIMELINESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SEARCH_TIMELINESS_PROMPT_ZH,
            ),
        ],
    },
)


class SearchTimelinessGrader(LLMGrader):
    """
    Search Timeliness Grader for Finance Domain

    Purpose:
        Evaluates the timeliness of stock search results by comparing two responses
        based on currency of information and temporal notation.

    What it evaluates:
        - Latest information: Use most recent public data, policies, and events
        - Temporal notation: Annotate time points for time-sensitive facts

    When to use:
        - Evaluating stock search result timeliness
        - Comparing currency of search responses
        - Assessing information freshness quality

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_SEARCH_TIMELINESS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_search.search_timeliness import SearchTimelinessGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SearchTimelinessGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="检索受关税政策影响的出口企业",
        ...     answer_1="申洲国际主要做纺织出口。",
        ...     answer_2="2024年3月，美国对华加征关税，申洲国际(纺织出口)受影响..."
        ... ))
        >>> print(result.rank)  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.ZH,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SearchTimelinessGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for search timeliness evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="search_timeliness",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock search timeliness by comparing two responses",
            model=model,
            template=template or DEFAULT_SEARCH_TIMELINESS_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs,
    ) -> GraderRank:
        """
        Evaluate two stock search responses for timeliness

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="查询最新的新能源汽车销量数据",
            ...     answer_1="新能源汽车销量增长。",
            ...     answer_2="2024年1月新能源汽车销量72.9万辆，同比增长78.8%..."
            ... )
        """
        try:
            result = await super()._aevaluate(
                query=query,
                answer_1=answer_1,
                answer_2=answer_2,
            )

            rank = result.rank
            reason = result.reason

            # Validate rank format
            if not isinstance(rank, list) or len(rank) != 2:
                logger.warning(f"Invalid rank format: {rank}, defaulting to [1, 2]")
                rank = [1, 2]

            # Ensure rank is either [1, 2] or [2, 1]
            if set(rank) != {1, 2}:
                logger.warning(f"Invalid rank values: {rank}, defaulting to [1, 2]")
                rank = [1, 2]

            return GraderRank(
                name=self.name,
                rank=rank,
                reason=reason,
                metadata={
                    "evaluation_type": "search_timeliness",
                    "criteria": ["latest_information", "temporal_notation"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating search timeliness: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SearchTimelinessGrader", "DEFAULT_SEARCH_TIMELINESS_TEMPLATE"]
