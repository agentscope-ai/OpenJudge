# -*- coding: utf-8 -*-
"""
Macro Analysis Grader for Finance Domain

Evaluates the quality of macroeconomic analysis by comparing two responses based on
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
MACRO_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Macroeconomic Analysis Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Macroeconomic Analysis**:
Criterion 1. Completeness of Macroeconomic Analysis: Assess whether the macroeconomic analysis in the answer is comprehensive, including but not limited to:
a. Clear transmission mechanism: Able to clearly articulate the causal chain from changes in macroeconomic indicators/concepts/events to market or economic outcomes (e.g., interest rate decline → financing cost reduction → investment demand increase → commodity demand increase)

Criterion 2. Depth of Macroeconomic Analysis: Assess whether the macroeconomic analysis is professional and in-depth, including but not limited to:
a. Multi-dimensional analysis:
    i. Time dimension: Reflect the temporal nature of financial event impacts, distinguish between short-term and long-term impacts, show phased evolution characteristics
    ii. Multi-level analysis: Construct a hierarchical analysis framework from macro to micro, must cover core driving dimensions affecting the market (policy, capital, fundamentals, external environment, industry, company, etc.)
b. Combination of quantitative and qualitative: Use quantitative analysis (models, historical regression, sensitivity analysis) where possible, supplemented by qualitative judgments for explanation
c. Case analysis: Can supplement analysis through historical cases

Criterion 3. Logical Consistency of Macroeconomic Analysis: Assess the logical consistency of macroeconomic analysis in the answer
a. Evidence support: Each conclusion needs to be clearly derived through evidence (numbers/descriptions, etc.), explaining the mechanism of action between variables
b. Each sub-argument should have a logical relationship with the financial question, not isolated listing, ensuring logical consistency
c. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency
</Rubrics>

<Steps>
1. First, determine whether there is macroeconomic analysis in the responses. A response with macro analysis is better;
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Steps>

<Constraints>
- **Objectivity**: Do not be influenced by the politeness or length of the response.
- **Focus**: Only evaluate based on the provided evaluation criteria.
- A response with macroeconomic analysis is better than one without.
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
MACRO_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**宏观分析质量**。

<评分标准>
请根据**宏观分析**评估回答：
评价标准1. 宏观分析的完整性：评估答案中的宏观分析是否全面，包括但不限于:
a. 传导机制清晰：能够清晰阐述从宏观指标/宏观概念/宏观事件变化到市场或经济结果的因果链条（例如：利率下降 → 融资成本降低 → 投资需求增加 → 大宗商品需求提升）

评价标准2. 宏观分析的深度：评估答案中的宏观分析是否专业有深度,包括但不限于:
a. 多维度分析：
    ⅰ. 时间维度：需体现金融事件影响的时序性，区分短期与长期影响，展示影响的阶段性演变特征
    ⅱ. 多层次分析：构建从宏观到微观的分层分析框架，必须覆盖影响市场的核心驱动维度（政策、资金、基本面、外部环境、行业、公司等）
b. 量化与定性结合：在可能的情况下使用量化分析（模型、历史回归、敏感性分析），并辅以定性判断进行解释
c. 案例分析：可以通过历史的案例进行补充分析

评价标准3. 宏观分析逻辑性：评估答案中宏观分析的逻辑性
a. 论据支撑：每个结论需通过论据(数字/描述等)明确推导，解释变量间作用机制
b. 各子论点应和金融问题有逻辑关系，非孤立罗列，确保逻辑自洽
c. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽
</评分标准>

<评估步骤>
1. 先判断回答中是否有宏观分析，能识别到的答案更好;
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<注意事项>
- **客观性**：不要被回答的礼貌程度或长度所影响。
- **聚焦**：仅根据提供的评估标准进行评估。
- 有宏观分析的回答优于没有的回答。
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
DEFAULT_MACRO_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=MACRO_ANALYSIS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=MACRO_ANALYSIS_PROMPT_ZH,
            ),
        ],
    },
)


class MacroAnalysisGrader(LLMGrader):
    """
    Macro Analysis Grader for Finance Domain

    Purpose:
        Evaluates the quality of macroeconomic analysis by comparing two responses based on
        completeness, depth, and logical consistency.

    What it evaluates:
        - Completeness: Clear transmission mechanisms and causal chains
        - Depth: Time dimension, multi-level analysis, quantitative/qualitative combination, cases
        - Logical Consistency: Evidence support, logical relationships

    When to use:
        - Evaluating macroeconomic analysis responses
        - Comparing quality of market impact analysis
        - Assessing depth of policy transmission analysis

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_MACRO_ANALYSIS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.macro_analysis.macro_analysis import MacroAnalysisGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = MacroAnalysisGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="分析降息对房地产市场的影响",
        ...     answer_1="降息会刺激房地产市场。",
        ...     answer_2="降息通过以下传导机制影响房地产：1)融资成本降低..."
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
        Initialize MacroAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for macro analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="macro_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate macroeconomic analysis quality by comparing two responses",
            model=model,
            template=template or DEFAULT_MACRO_ANALYSIS_TEMPLATE,
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
        Evaluate two macroeconomic analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析当前中国宏观经济形势",
            ...     answer_1="经济形势总体稳定。",
            ...     answer_2="当前中国经济呈现稳中向好态势：GDP增速5.2%，CPI温和..."
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
                    "evaluation_type": "macro_analysis",
                    "criteria": ["completeness", "depth", "logical_consistency"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating macro analysis: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["MacroAnalysisGrader", "DEFAULT_MACRO_ANALYSIS_TEMPLATE"]
