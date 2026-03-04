# -*- coding: utf-8 -*-
"""
Stock Risk Analysis Grader for Finance Domain

Evaluates the quality of stock risk analysis by comparing two responses based on
completeness, depth, and logical rigor. This grader is specific to stock analysis
and includes company-specific risks in addition to market and industry risks.
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
STOCK_RISK_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Stock Risk Analysis Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Stock Risk Analysis**:
Criterion 1. Completeness of Risk Analysis: Assess whether the risk analysis in the response is comprehensive, including but not limited to:
a. Policy risks (such as tightening environmental policies, tax policy changes)
b. Industry risks (such as commodity price fluctuations in mining, intensified competition in express delivery)
c. Market risks (such as rising interest rates affecting valuations, exchange rate fluctuations affecting overseas business)
d. Company risks (such as management changes, insufficient capacity utilization, financial fraud risks)

Criterion 2. Depth of Risk Analysis: Assess whether the risk analysis is quantified and closely related to company business, including but not limited to:
a. Quantifying the impact of risks on financial indicators (e.g., "A 10% drop in copper prices will reduce Zijin Mining's net profit by 8%")
b. Probability of risk occurrence (e.g., "High probability of tightening environmental policies due to national carbon neutrality targets")
c. Company or industry-specific risks (e.g., "Mining companies may face safety accidents")

Criterion 3. Logical Rigor of Risk Analysis: Assess whether the risk analysis has a logical transmission path (e.g., "Oil price increase → Express delivery transportation cost increase → Gross margin decline → Net profit decline")
</Rubrics>

<Steps>
1. First, determine whether there is risk analysis in the responses. A response with risk analysis is better than one without; if both responses lack risk analysis, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Steps>

<Constraints>
- **Objectivity**: Do not be influenced by the politeness or length of the response.
- **Focus**: Only evaluate based on the provided evaluation criteria.
- A response with risk analysis is better than one without.
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
STOCK_RISK_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**股票风险分析质量**。

<评分标准>
请根据**股票风险分析**评估回答：
评价标准1. 风险分析完整性：评估回答中的风险分析是否全面，包含但不限于：
a. 政策风险（如环保政策收紧、税收政策变化）
b. 行业风险（如矿业的大宗商品价格波动、快递业的行业竞争加剧）
c. 市场风险（如利率上升影响估值、汇率波动影响海外业务）
d. 公司风险（如管理层变动、产能利用率不足、财务造假风险）

评价标准2. 风险分析深度：评估回答中的风险分析是否量化以及是否和公司业务等紧密相关，包括但不限于：
a. 量化风险对财务指标的影响（如 "铜价下跌 10%，紫金矿业净利润将减少 8%"）
b. 风险发生的概率（如 "环保政策收紧概率高，因国家明确双碳目标"）
c. 公司或者所在行业特有的风险(如 "矿产公司可能有安全事故")

评价标准3. 风险分析逻辑严密性：评估回答中的风险分析有无逻辑传导路径(如"油价上涨→快递运输成本增加→毛利率下降→净利润下滑")
</评分标准>

<评估步骤>
1. 先判断回答中有风险分析，有风险分析的回答优于没有风险分析的回答；如果两个回答都没有风险分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由；
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<注意事项>
- **客观性**：不要被回答的礼貌程度或长度所影响。
- **聚焦**：仅根据提供的评估标准进行评估。
- 有风险分析的回答优于没有风险分析的回答。
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
DEFAULT_STOCK_RISK_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=STOCK_RISK_ANALYSIS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=STOCK_RISK_ANALYSIS_PROMPT_ZH,
            ),
        ],
    },
)


class StockRiskAnalysisGrader(LLMGrader):
    """
    Stock Risk Analysis Grader for Finance Domain

    Purpose:
        Evaluates the quality of stock risk analysis by comparing two responses based on
        completeness, depth, and logical rigor. Includes company-level risks.

    What it evaluates:
        - Completeness: Coverage of policy, industry, market, and company risks
        - Depth: Quantification of impacts, probability assessment, company-specific risks
        - Logical Rigor: Clear transmission paths showing cause and effect

    When to use:
        - Evaluating stock-specific risk analysis responses
        - Comparing quality of investment risk assessment
        - Assessing depth of company risk quantification

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_STOCK_RISK_ANALYSIS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_analysis.stock_risk_analysis import StockRiskAnalysisGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = StockRiskAnalysisGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="分析宁德时代的投资风险",
        ...     answer_1="存在市场波动风险。",
        ...     answer_2="风险：1)政策风险-补贴退坡；2)竞争风险-比亚迪挤压；3)公司风险-产能扩张过快..."
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
        Initialize StockRiskAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for stock risk analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="stock_risk_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock risk analysis quality by comparing two responses",
            model=model,
            template=template or DEFAULT_STOCK_RISK_ANALYSIS_TEMPLATE,
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
        Evaluate two stock risk analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析比亚迪股票的风险",
            ...     answer_1="存在市场竞争风险。",
            ...     answer_2="主要风险：1)补贴退坡风险 2)原材料价格波动 3)技术迭代..."
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
                    "evaluation_type": "stock_risk_analysis",
                    "criteria": ["completeness", "depth", "logical_rigor"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating stock risk analysis: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["StockRiskAnalysisGrader", "DEFAULT_STOCK_RISK_ANALYSIS_TEMPLATE"]
