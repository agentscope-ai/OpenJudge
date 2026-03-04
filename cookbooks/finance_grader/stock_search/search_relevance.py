# -*- coding: utf-8 -*-
"""
Search Relevance Grader for Finance Domain

Evaluates the relevance of stock search results by comparing two responses
based on alignment with query requirements and precision matching.
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
SEARCH_RELEVANCE_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Search Relevance Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Search Relevance**:
Criterion 1. Search Relevance: Assess the relevance of stocks retrieved in the answer to the question, including but not limited to:
a. Responses should strictly focus on the core requirements of the question (industry scope, geographic conditions, business characteristics, etc.), without introducing irrelevant industries or companies.
b. Precise matching: Search results must precisely match the geographic, industry, and business characteristics specified in the question. For example, "domestically listed export companies with production capacity in Vietnam" must simultaneously meet "domestic listing + export + Vietnam capacity" conditions.
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
SEARCH_RELEVANCE_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**检索相关性质量**。

<评分标准>
请根据**检索相关性**评估回答：
评价标准1. 检索相关性：评估答案中检索到的股票和问题的相关性，包括但不限于：
a. 回答应严格围绕问题核心要求（行业范围、地域条件、业务特征等），不引入无关行业或公司。
b. 精准匹配：检索结果需与问题限定的地域、行业、业务特性精准匹配，例如"越南有产能布局的国内上市出口公司"需同时满足"国内上市 + 出口 + 越南产能"条件。
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
DEFAULT_SEARCH_RELEVANCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SEARCH_RELEVANCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SEARCH_RELEVANCE_PROMPT_ZH,
            ),
        ],
    },
)


class SearchRelevanceGrader(LLMGrader):
    """
    Search Relevance Grader for Finance Domain

    Purpose:
        Evaluates the relevance of stock search results by comparing two responses
        based on alignment with core requirements and precise matching of conditions.

    What it evaluates:
        - Focus on core requirements: Industry scope, geography, business characteristics
        - Precise matching: All specified conditions must be met simultaneously

    When to use:
        - Evaluating stock search result relevance
        - Comparing precision of search responses
        - Assessing query-result alignment quality

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_SEARCH_RELEVANCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_search.search_relevance import SearchRelevanceGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SearchRelevanceGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="检索在越南有产能布局的国内上市出口公司",
        ...     answer_1="阿里巴巴是国内上市公司。",
        ...     answer_2="申洲国际：国内上市，纺织出口，越南有产能；立讯精密..."
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
        Initialize SearchRelevanceGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for search relevance evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="search_relevance",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock search relevance by comparing two responses",
            model=model,
            template=template or DEFAULT_SEARCH_RELEVANCE_TEMPLATE,
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
        Evaluate two stock search responses for relevance

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="搜索光伏行业龙头股",
            ...     answer_1="隆基绿能是光伏企业。",
            ...     answer_2="光伏龙头：隆基绿能(硅片)、通威股份(硅料)、阳光电源(逆变器)..."
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
                    "evaluation_type": "search_relevance",
                    "criteria": ["core_alignment", "precise_matching"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating search relevance: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SearchRelevanceGrader", "DEFAULT_SEARCH_RELEVANCE_TEMPLATE"]
