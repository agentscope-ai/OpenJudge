# -*- coding: utf-8 -*-
"""
Underlying Comparison Grader for Finance Domain

Evaluates the quality of underlying (company/stock) comparison analysis by comparing
two responses based on completeness, depth, and logical reasoning.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# Chinese Prompt (Primary language for finance domain)
UNDERLYING_COMPARISON_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答对标的对比分析的专业性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 先判断回答中是否有行业标的对比分析，有标的对比分析的回答优于没有的标的对比分析的回答；如果两个回答都没有标的对比分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 标的对比分析的完整性：评估答案中的行业标的对比分析是否全面，包括但不限于:
a. 覆盖代表性标的：覆盖行业内主要参与者，包括不同业务模式和战略定位的代表企业
b. 覆盖标的多维度信息：包括但不限于业务分析、财务分析、估值分析、风险分析

评价标准2. 标的对比分析的深度：评估答案中能否挖掘出标的核心差异,包括但不限于:
a. 差异化优势识别清晰：深入分析各企业差异化竞争策略，揭示竞争优势来源和可持续性
b. 标的横向可比：使用统一可比指标进行对比（如ROE、净息差、不良率等），避免混用不同单位或口径
c. 分析严谨性：提供可验证的量化指标支撑竞争优势分析，避免主观定性描述

评价标准3. 标的对比分析逻辑性：评估答案中标的对比分析的逻辑性
a. 论据支撑：每个结论需通过论据(数字/描述等)明确推导，解释变量间作用机制
b. 各子论点应和金融问题有逻辑关系，非孤立罗列，确保逻辑自洽
c. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽
</评估标准>

<金融问题>
{query}
</金融问题>

<回答1>
{answer_1}
</回答1>

<回答2>
{answer_2}
</回答2>

# 评分指令
请根据上述评估标准，比较两个回答的质量。
- 如果回答1更好：rank = [1, 2]
- 如果回答2更好：rank = [2, 1]

请按以下结构化 JSON 格式提供你的评估：
{{
    "rank": <[1, 2] 或 [2, 1]>,
    "reason": "<详细解释你的评估理由，包括在各个评估标准下的表现对比>"
}}

JSON:
"""

# English Prompt
UNDERLYING_COMPARISON_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of underlying (company/stock) comparison analysis in financial assistants' responses. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. First, determine whether there is underlying comparison analysis in the responses. A response with underlying comparison analysis is better than one without; if both responses lack it, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Completeness of Underlying Comparison Analysis: Assess whether the underlying comparison analysis in the answer is comprehensive, including but not limited to:
a. Coverage of representative underlyings: Cover major participants in the industry, including representative companies with different business models and strategic positions
b. Multi-dimensional information coverage: Including but not limited to business analysis, financial analysis, valuation analysis, risk analysis

Criterion 2. Depth of Underlying Comparison Analysis: Assess whether the answer can uncover core differences between underlyings, including but not limited to:
a. Clear identification of differentiated advantages: In-depth analysis of each company's differentiated competitive strategies, revealing sources and sustainability of competitive advantages
b. Horizontal comparability of underlyings: Use unified comparable indicators for comparison (such as ROE, net interest margin, NPL ratio, etc.), avoid mixing different units or calibers
c. Analytical rigor: Provide verifiable quantitative indicators to support competitive advantage analysis, avoid subjective qualitative descriptions

Criterion 3. Logical Consistency of Underlying Comparison Analysis: Assess the logical consistency of underlying comparison analysis in the answer
a. Evidence support: Each conclusion needs to be clearly derived through evidence (numbers/descriptions, etc.), explaining the mechanism of action between variables
b. Each sub-argument should have a logical relationship with the financial question, not isolated listing, ensuring logical consistency
c. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency
</Evaluation Criteria>

<Financial Question>
{query}
</Financial Question>

<Answer 1>
{answer_1}
</Answer 1>

<Answer 2>
{answer_2}
</Answer 2>

# Scoring Instructions
Based on the above evaluation criteria, compare the quality of the two answers.
- If Answer 1 is better: rank = [1, 2]
- If Answer 2 is better: rank = [2, 1]

Please provide your evaluation in the following structured JSON format:
{{
    "rank": <[1, 2] or [2, 1]>,
    "reason": "<detailed explanation of your evaluation reasoning, including performance comparison under each evaluation criterion>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_UNDERLYING_COMPARISON_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(UNDERLYING_COMPARISON_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(UNDERLYING_COMPARISON_PROMPT_ZH),
            ),
        ],
    },
)


class UnderlyingComparisonGrader(LLMGrader):
    """
    Underlying Comparison Grader for Finance Domain

    Evaluates the quality of underlying (company/stock) comparison analysis by comparing
    two responses based on completeness (representative coverage, multi-dimensional info),
    depth (differentiation, comparability, rigor), and logical consistency.

    Evaluation Criteria:
    1. Completeness: Coverage of representative underlyings and multi-dimensional information
    2. Depth: Clear differentiation, horizontal comparability, analytical rigor
    3. Logical Consistency: Evidence support, logical relationships between arguments

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.ZH for Chinese)

    Example:
        >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
        >>> from rm_gallery.core.models.schema.prompt_template import LanguageEnum
        >>>
        >>> model = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = UnderlyingComparisonGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="对比分析宁德时代和比亚迪",
        ...     answer_1="两家都是动力电池龙头企业。",
        ...     answer_2="宁德时代ROE 25%，专注电池；比亚迪ROE 18%，垂直整合..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_UNDERLYING_COMPARISON_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize UnderlyingComparisonGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for underlying comparison evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="underlying_comparison",
            mode=GraderMode.LISTWISE,
            description="Evaluate underlying comparison analysis quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_UNDERLYING_COMPARISON_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two underlying comparison analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better
        """
        try:
            result = await super().aevaluate(
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

        except Exception as e:
            logger.error(f"Error evaluating underlying comparison: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "underlying_comparison",
            "criteria": ["completeness", "depth", "logical_consistency"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
