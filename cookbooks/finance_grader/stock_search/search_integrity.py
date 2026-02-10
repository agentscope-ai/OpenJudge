# -*- coding: utf-8 -*-
"""
Search Integrity Grader for Finance Domain

Evaluates the integrity and completeness of stock search results by comparing
two responses based on coverage and information completeness.
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
SEARCH_INTEGRITY_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Search Integrity Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Search Integrity**:
Criterion 1. Search Integrity: Assess the integrity of stock information retrieved in the answer, including but not limited to:
a. Comprehensive coverage: While ensuring accuracy, should cover as many qualifying major companies as possible, rather than listing only one or a few examples.
b. If information is incomplete, should clearly state missing parts and suggest possible sources or query recommendations.
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
SEARCH_INTEGRITY_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**检索完整性质量**。

<评分标准>
请根据**检索完整性**评估回答：
评价标准1. 检索完整性：评估答案中检索到的股票信息的完整性，包括但不限于：
a. 覆盖全面：在确保准确性的前提下，应尽可能覆盖符合条件的主要公司，而非只列举单个或少数示例。
b. 若信息不全，应明确说明缺失部分并提示可能的来源或查询建议。
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
DEFAULT_SEARCH_INTEGRITY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=SEARCH_INTEGRITY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=SEARCH_INTEGRITY_PROMPT_ZH,
            ),
        ],
    },
)


class SearchIntegrityGrader(LLMGrader):
    """
    Search Integrity Grader for Finance Domain

    Purpose:
        Evaluates the integrity and completeness of stock search results by comparing
        two responses based on comprehensive coverage and information completeness.

    What it evaluates:
        - Comprehensive coverage: Cover major qualifying companies, not just examples
        - Information completeness: State missing information and suggest sources

    When to use:
        - Evaluating stock search result completeness
        - Comparing coverage of search responses
        - Assessing information retrieval quality

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_SEARCH_INTEGRITY_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_search.search_integrity import SearchIntegrityGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SearchIntegrityGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="检索新能源汽车行业的上市公司",
        ...     answer_1="比亚迪是新能源汽车龙头。",
        ...     answer_2="新能源汽车上市公司包括：比亚迪、宁德时代、理想汽车..."
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
        Initialize SearchIntegrityGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for search integrity evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="search_integrity",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock search integrity and completeness by comparing two responses",
            model=model,
            template=template or DEFAULT_SEARCH_INTEGRITY_TEMPLATE,
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
        Evaluate two stock search responses for integrity

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="搜索新能源汽车相关股票",
            ...     answer_1="比亚迪是新能源汽车股票。",
            ...     answer_2="新能源汽车板块主要标的：比亚迪、宁德时代、长城汽车..."
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
                    "evaluation_type": "search_integrity",
                    "criteria": ["comprehensive_coverage", "information_completeness"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating search integrity: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SearchIntegrityGrader", "DEFAULT_SEARCH_INTEGRITY_TEMPLATE"]
