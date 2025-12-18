# -*- coding: utf-8 -*-
"""
Industry Characteristics Analysis Grader for Finance Domain

Evaluates the quality of industry characteristics analysis by comparing two responses
based on completeness, depth, and logical reasoning.
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
CHARACTERISTICS_ANALYSIS_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答对行业分析的专业度。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 先判断回答中是否有行业特性分析，有行业特性分析的回答优于没有的回答；如果两个回答都没有，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 行业特性分析的完整性：评估答案中的行业特性分析是否全面，包括但不限于:
a. 产业链结构分析：上下游环节、主要产品与服务类型、成本结构、关键原材料来源
b. 行业驱动因素剖析：需求端（政策、消费趋势、宏观经济周期）、供给端（产能扩张、进入壁垒、技术迭代）
c. 周期属性与关键变量：准确把握行业所处发展阶段及特征，识别关键趋势和转折点
d. 监管与政策影响：政策倾斜、税收、环保、进出口限制等对未来格局的影响
e. 竞争格局描述：CRn（行业集中度）、主要玩家市占率变化、格局演化趋势

评价标准2. 行业特性分析的深度：评估答案中能否挖掘出行业特性,包括但不限于:
a. 对于行业相关分析(产业链结构、行业驱动因素、周期属性、监管与政策影响、竞争格局描述)不是泛泛而谈，有论据支撑
b. 行业分析要给出整体结论，不能只罗列特性

评价标准3. 行业特性分析逻辑性：评估答案中行业特性分析的逻辑性
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
CHARACTERISTICS_ANALYSIS_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of financial assistants' industry analysis. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. First, determine whether there is industry characteristics analysis in the responses. A response with industry characteristics analysis is better than one without; if both responses lack it, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Completeness of Industry Characteristics Analysis: Assess whether the industry characteristics analysis in the answer is comprehensive, including but not limited to:
a. Industrial chain structure analysis: upstream and downstream links, main products and service types, cost structure, key raw material sources
b. Analysis of industry driving factors: demand side (policies, consumption trends, macroeconomic cycles), supply side (capacity expansion, entry barriers, technological iteration)
c. Cycle attributes and key variables: accurately grasp the development stage and characteristics of the industry, identify key trends and turning points
d. Regulatory and policy impacts: policy preferences, taxation, environmental protection, import and export restrictions, etc., on future landscape
e. Competitive landscape description: CRn (industry concentration), major players' market share changes, landscape evolution trends

Criterion 2. Depth of Industry Characteristics Analysis: Assess whether the answer can uncover industry characteristics, including but not limited to:
a. Industry-related analysis (industrial chain structure, driving factors, cycle attributes, regulatory and policy impacts, competitive landscape) is not superficial but supported by evidence
b. Industry analysis should provide overall conclusions, not just list characteristics

Criterion 3. Logical Consistency of Industry Characteristics Analysis: Assess the logical consistency of industry characteristics analysis in the answer
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
DEFAULT_CHARACTERISTICS_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CHARACTERISTICS_ANALYSIS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CHARACTERISTICS_ANALYSIS_PROMPT_ZH),
            ),
        ],
    },
)


class CharacteristicsAnalysisGrader(LLMGrader):
    """
    Industry Characteristics Analysis Grader for Finance Domain

    Evaluates the quality of industry characteristics analysis by comparing two responses
    based on completeness (industrial chain, driving factors, cycle attributes, regulation,
    competition), depth (evidence support, overall conclusions), and logical consistency.

    Evaluation Criteria:
    1. Completeness: Coverage of industrial chain, driving factors, cycle, regulation, competition
    2. Depth: Evidence-based analysis with overall conclusions
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
        >>> grader = CharacteristicsAnalysisGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="分析新能源汽车行业的特性",
        ...     answer_1="新能源汽车行业发展迅速。",
        ...     answer_2="新能源汽车行业产业链完整，上游包括锂矿、电池材料..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_CHARACTERISTICS_ANALYSIS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize CharacteristicsAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for characteristics analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="characteristics_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate industry characteristics analysis quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_CHARACTERISTICS_ANALYSIS_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two industry characteristics analysis responses

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
            logger.error(f"Error evaluating characteristics analysis: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "characteristics_analysis",
            "criteria": ["completeness", "depth", "logical_consistency"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
