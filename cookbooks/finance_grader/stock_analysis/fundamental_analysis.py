# -*- coding: utf-8 -*-
"""
Fundamental Analysis Grader for Finance Domain

Evaluates the quality of fundamental analysis by comparing two responses based on
completeness, depth, and logical reasoning.
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
FUNDAMENTAL_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Fundamental Analysis Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Fundamental Analysis**:
Criterion 1. Completeness of Fundamental Analysis: Assess whether the fundamental analysis in the answer is comprehensive, including but not limited to:
a. Company business analysis, including but not limited to: core business analysis (company's core business, products, services), revenue composition analysis (revenue composition and stability/growth of each segment), business model interpretation (profit model/cost structure/pricing power);
b. Industry landscape and competitive analysis, including but not limited to: industry cycle and trends (reveal industry cycle stage, supply-demand landscape, and policy environment impact mechanisms), industry competitive landscape (identify major competitors and comparative analysis), individual stock competitiveness analysis (consider competitive advantages within the industry landscape);
c. Financial health status, including but not limited to: profitability analysis (analyze trends of key indicators such as gross margin, net margin, ROE, ROIC, etc.), asset-liability structure: analyze debt ratio, short-term solvency (current ratio, quick ratio), capital structure stability;

Criterion 2. Depth of Fundamental Analysis: Assess whether the answer includes company-specific characteristics, including but not limited to:
a. Key business drivers: core driving factors behind the company's operational performance (technological advantages, channels, brand, etc.);
b. Company's competitive advantages in the industry, including but not limited to: cost advantages, technological advantages;
c. Company's growth analysis;
d. Other content that can reflect company characteristics;

Criterion 3. Logical Consistency of Fundamental Analysis: Assess the answer's
a. Evidence support: Each conclusion needs to be clearly derived through evidence (numbers/descriptions, etc.), explaining the mechanism of action between variables
b. Each sub-argument should have a logical relationship with the financial question, not isolated listing, ensuring logical consistency
c. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency
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
FUNDAMENTAL_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**基本面分析质量**。

<评分标准>
请根据**基本面分析**评估回答：
评价标准1. 基本面分析的完整性：评估答案中的基本面分析是否全面，包括但不限于:
a. 公司的业务分析，包括但不限于：主营业务分析(公司核心业务、产品、服务)、收入构成分析(收入构成以及各板块收入的稳定性/增长性)、商业模式解读(盈利模式/成本结构/议价能力);
b. 行业赛道与竞争格局分析，包括但不限于：行业周期与趋势(揭示行业所处周期阶段、供给需求格局及政策环境影响机制)、行业竞争格局(识别主要竞争对手并做对比分析)、个股竞争力分析(考虑行业竞争格局中个股的竞争优势);
c. 财务健康情况，包括但不限于：盈利能力分析(需要分析关键指标(毛利率、净利率、ROE、ROIC等)的变化趋势)、资产负债结构：需要分析负债率、短期偿债能力（流动比率、速动比率）、资本结构稳定性;

评价标准2. 基本面分析的深度：评估答案中是否包含公司的特质,包括但不限于:
a. 公司关键业务驱动因素：公司经营表现背后的核心驱动因素(技术优势、渠道、品牌等);
b. 公司在本行业的竞争优势，包括但不限于：成本优势、技术优势;
c. 公司的成长性分析;
d. 其他能够体现公司特质的其他内容;

评价标准3. 基本面分析逻辑性：评估答案的
a. 论据支撑：每个结论需通过论据(数字/描述等)明确推导，解释变量间作用机制
b. 各子论点应和金融问题有逻辑关系，非孤立罗列，确保逻辑自洽
c. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽
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
DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=FUNDAMENTAL_ANALYSIS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=FUNDAMENTAL_ANALYSIS_PROMPT_ZH,
            ),
        ],
    },
)


class FundamentalAnalysisGrader(LLMGrader):
    """
    Fundamental Analysis Grader for Finance Domain

    Purpose:
        Evaluates the quality of fundamental analysis by comparing two responses based on
        completeness, depth, and logical consistency.

    What it evaluates:
        - Completeness: Business analysis, industry/competition, financial health
        - Depth: Key business drivers, competitive advantages, growth analysis, company specifics
        - Logical Consistency: Evidence support, logical relationships

    When to use:
        - Evaluating stock fundamental analysis responses
        - Comparing quality of company research
        - Assessing depth of financial analysis

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_analysis.fundamental_analysis import FundamentalAnalysisGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = FundamentalAnalysisGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="分析宁德时代的基本面",
        ...     answer_1="宁德时代是动力电池龙头。",
        ...     answer_2="宁德时代主营动力电池，ROE 25%，市占率33%..."
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
        Initialize FundamentalAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for fundamental analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="fundamental_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate fundamental analysis quality by comparing two responses",
            model=model,
            template=template or DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE,
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
        Evaluate two fundamental analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析贵州茅台的基本面",
            ...     answer_1="茅台是白酒龙头企业。",
            ...     answer_2="茅台基本面优秀：毛利率91%，ROE 30%，现金流充沛..."
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
                    "evaluation_type": "fundamental_analysis",
                    "criteria": ["completeness", "depth", "logical_consistency"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating fundamental analysis: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["FundamentalAnalysisGrader", "DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE"]
