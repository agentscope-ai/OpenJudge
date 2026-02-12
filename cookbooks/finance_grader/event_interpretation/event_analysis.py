# -*- coding: utf-8 -*-
"""
Event Analysis Grader for Finance Domain

Evaluates the quality of financial event analysis by comparing two responses
based on comprehensiveness, depth, and logical reasoning.
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
EVENT_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Event Analysis Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Event Analysis**:
Criterion 1. Comprehensiveness of Event Analysis: Assess whether the event analysis in the answer is comprehensive, including but not limited to:
a. Event background or causes: Whether the macro, industry, or historical background of the event is adequately presented.
b. Event impact mechanism analysis: Accurately explain the impact and transmission path of the event on the market, industry, or company

Criterion 2. Depth of Event Analysis: Assess whether the event analysis in the answer is in-depth, including but not limited to:
a. Event impact mechanism analysis: Accurately explain the transmission path between relevant financial variables, deeply analyze how the event affects valuation mechanisms, capital flows, or market pricing logic
b. Multi-dimensional analysis of event impact mechanisms, including but not limited to:
    i. Time dimension: Reflect the temporal nature of financial event impacts, distinguish between short-term and long-term impacts, and show the phased evolution characteristics of the impact
    ii. Multi-level analysis: Construct a hierarchical analysis framework from macro to micro, must cover core driving dimensions affecting the market (policy, capital, fundamentals, external environment, industry, company, etc.)
c. Combination of quantitative and qualitative: Use quantitative analysis (models, historical regression, sensitivity analysis) where possible, supplemented by qualitative judgments for explanation
d. Case analysis: Can supplement analysis through historical cases

Criterion 3. Logical Consistency of Event Analysis: Assess the logical consistency of the event analysis in the answer
a. Evidence support: Each conclusion needs to be clearly derived through evidence (numbers/descriptions, etc.), explaining the mechanism of action between variables
b. Each sub-argument should have a logical relationship with the financial question, not isolated listing, ensuring logical consistency
c. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency
</Rubrics>

<Steps>
1. First, determine whether there is event analysis in the responses. A response with event analysis is better than one without; if both responses lack event analysis, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Steps>

<Constraints>
- **Objectivity**: Do not be influenced by the politeness or length of the response.
- **Focus**: Only evaluate based on the provided evaluation criteria.
- A response with event analysis is better than one without event analysis.
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
EVENT_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**事件分析质量**。

<评分标准>
请根据**事件分析**评估回答：
评价标准1. 事件分析的完整性：评估答案中的事件分析是否全面，包括但不限于:
a. 事件的背景或者起因：是否充分地呈现了事件发生的宏观、行业或历史背景。
b. 事件影响机制分析：需准确解释事件对市场或者行业或者公司的影响和传导路径

评价标准2. 事件分析的深度：评估答案中的事件分析是否深入，包括但不限于:
a. 事件影响机制分析：需准确解释相关金融变量之间的传导路径，深入剖析事件如何影响估值机制、资金流向或市场定价逻辑
b. 事件影响机制多维度分析，包括但不限于：
    ⅰ. 时间维度：需体现金融事件影响的时序性，区分短期与长期影响，展示影响的阶段性演变特征
    ⅱ. 多层次分析：构建从宏观到微观的分层分析框架，必须覆盖影响市场的核心驱动维度（政策、资金、基本面、外部环境、行业、公司等）
c. 量化与定性结合：在可能的情况下使用量化分析（模型、历史回归、敏感性分析），并辅以定性判断进行解释
d. 案例分析：可以通过历史的案例进行补充分析

评价标准3. 事件分析逻辑性：评估答案中事件分析的逻辑性
a. 论据支撑：每个结论需通过论据(数字/描述等)明确推导，解释变量间作用机制
b. 各子论点应和金融问题有逻辑关系，非孤立罗列，确保逻辑自洽
c. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽
</评分标准>

<评估步骤>
1. 先判断回答中是否有事件分析，有事件分析的回答优于没有事件分析的回答；如果两个回答都没有事件分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<注意事项>
- **客观性**：不要被回答的礼貌程度或长度所影响。
- **聚焦**：仅根据提供的评估标准进行评估。
- 有事件分析的回答优于没有事件分析的回答。
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
DEFAULT_EVENT_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=EVENT_ANALYSIS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=EVENT_ANALYSIS_PROMPT_ZH,
            ),
        ],
    },
)


class EventAnalysisGrader(LLMGrader):
    """
    Event Analysis Grader for Finance Domain

    Purpose:
        Evaluates the quality of financial event analysis by comparing two responses based on
        comprehensiveness, depth, and logical reasoning.

    What it evaluates:
        - Comprehensiveness: Whether the event analysis covers background, causes, and impact mechanisms
        - Depth: Multi-dimensional analysis including time dimension, multi-level framework,
          quantitative/qualitative combination, and case studies
        - Logical Consistency: Evidence support, logical relationships between arguments and conclusions

    When to use:
        - Evaluating financial event analysis responses
        - Comparing quality of event interpretation
        - Assessing depth of market impact analysis

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_EVENT_ANALYSIS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.event_interpretation.event_analysis import EventAnalysisGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = EventAnalysisGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="分析美联储加息对中国股市的影响",
        ...     answer_1="美联储加息会导致资金回流美国，对中国股市形成压力。",
        ...     answer_2="美联储加息通过多个传导路径影响中国股市：1)资金面：加息导致美元走强..."
        ... ))
        >>> print(result.rank)  # [2, 1] means answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.ZH,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize EventAnalysisGrader.

        Args:
            model: The chat model to use for evaluation, either as a BaseChatModel instance or config dict
            template: The prompt template for event analysis evaluation.
                     Defaults to DEFAULT_EVENT_ANALYSIS_TEMPLATE.
            language: The language for the evaluation prompt. Defaults to LanguageEnum.ZH (Chinese).
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="event_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate financial event analysis quality by comparing two responses",
            model=model,
            template=template or DEFAULT_EVENT_ANALYSIS_TEMPLATE,
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
        Evaluate two financial event analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析美联储加息对中国股市的影响",
            ...     answer_1="美联储加息会导致资金回流美国。",
            ...     answer_2="美联储加息通过多个传导路径影响中国股市..."
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
                    "evaluation_type": "event_analysis",
                    "criteria": ["comprehensiveness", "depth", "logical_consistency"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating event analysis: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["EventAnalysisGrader", "DEFAULT_EVENT_ANALYSIS_TEMPLATE"]
