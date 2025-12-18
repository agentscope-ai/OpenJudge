# -*- coding: utf-8 -*-
"""
Event Analysis Grader for Finance Domain

Evaluates the quality of financial event analysis by comparing two responses
based on comprehensiveness, depth, and logical reasoning.
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
EVENT_ANALYSIS_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答对事件解读的专业性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 先判断回答中是否有事件分析，有事件分析的回答优于没有事件分析的回答；如果两个回答都没有事件分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
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
EVENT_ANALYSIS_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of financial assistants' responses in event interpretation. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. First, determine whether there is event analysis in the responses. A response with event analysis is better than one without; if both responses lack event analysis, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Comprehensiveness of Event Analysis: Assess whether the event analysis in the answer is comprehensive, including but not limited to:
a. Event background or causes: Whether the macro, industry, or historical background of the event is adequately presented.
b. Event impact mechanism analysis: Accurately explain the impact and transmission path of the event on the market, industry, or company

Criterion 2. Depth of Event Analysis: Assess whether the event analysis in the answer is in-depth, including but not limited to:
a. Event impact mechanism analysis: Accurately explain the transmission path between relevant financial variables, deeply analyze how the event affects valuation mechanisms, capital flows, or market pricing logic
b. Multi-dimensional analysis of event impact mechanisms, including but not limited to:
    ⅰ. Time dimension: Reflect the temporal nature of financial event impacts, distinguish between short-term and long-term impacts, and show the phased evolution characteristics of the impact
    ⅱ. Multi-level analysis: Construct a hierarchical analysis framework from macro to micro, must cover core driving dimensions affecting the market (policy, capital, fundamentals, external environment, industry, company, etc.)
c. Combination of quantitative and qualitative: Use quantitative analysis (models, historical regression, sensitivity analysis) where possible, supplemented by qualitative judgments for explanation
d. Case analysis: Can supplement analysis through historical cases

Criterion 3. Logical Consistency of Event Analysis: Assess the logical consistency of the event analysis in the answer
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
DEFAULT_EVENT_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(EVENT_ANALYSIS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(EVENT_ANALYSIS_PROMPT_ZH),
            ),
        ],
    },
)


class EventAnalysisGrader(LLMGrader):
    """
    Event Analysis Grader for Finance Domain

    Evaluates the quality of financial event analysis by comparing two responses based on
    comprehensiveness, depth, and logical reasoning. This is a pairwise comparison grader
    that ranks two responses.

    Evaluation Criteria:
    1. Comprehensiveness: Whether the event analysis covers background, causes, and impact mechanisms
    2. Depth: Multi-dimensional analysis including time dimension, multi-level framework,
       quantitative/qualitative combination, and case studies
    3. Logical Consistency: Evidence support, logical relationships between arguments and conclusions

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
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = EventAnalysisGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="分析美联储加息对中国股市的影响",
        ...     answer_1="美联储加息会导致资金回流美国，对中国股市形成压力。",
        ...     answer_2="美联储加息通过多个传导路径影响中国股市：1)资金面：加息导致美元走强..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] means answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_EVENT_ANALYSIS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize EventAnalysisGrader.

        Args:
            model: The chat model to use for evaluation, either as a BaseChatModel instance or config dict
            template: The prompt template for event analysis evaluation.
                     Defaults to DEFAULT_EVENT_ANALYSIS_TEMPLATE.
            language: The language for the evaluation prompt. Defaults to LanguageEnum.ZH (Chinese).
        """
        super().__init__(
            name="event_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate financial event analysis quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_EVENT_ANALYSIS_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
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
            ...     answer_1="简短回答...",
            ...     answer_2="详细分析，包含传导机制、时间维度等..."
            ... )
            >>> print(result.rank)  # [2, 1] if answer_2 is better
            >>> print(result.reason)  # Detailed explanation
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
            logger.error(f"Error evaluating event analysis: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "event_analysis",
            "criteria": ["comprehensiveness", "depth", "logical_consistency"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
