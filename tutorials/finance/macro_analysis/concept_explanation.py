# -*- coding: utf-8 -*-
"""
Concept Explanation Grader for Finance Domain

Evaluates the quality of macroeconomic concept explanations by comparing two responses
based on definition clarity and historical context.
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
CONCEPT_EXPLANATION_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答宏观分析问题的专业性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由;
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 定义清晰准确：对宏观指标（如利率、CPI、GDP、社零、PMI等）、宏观概念（如实际利率、潜在名义增速、收益率倒挂等）、宏观事件（如央行降息、股指突破关键点位等）给出清晰准确的的定义

评价标准2. 背景与历史对比：在解释时适当给出历史区间的变化趋势或历史均值，帮助理解指标当前所处的位置和意义
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
CONCEPT_EXPLANATION_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of financial assistants' responses to macroeconomic analysis questions. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Clear and Accurate Definition: Provide clear and accurate definitions for macroeconomic indicators (such as interest rates, CPI, GDP, retail sales, PMI, etc.), macroeconomic concepts (such as real interest rates, potential nominal growth rate, yield curve inversion, etc.), and macroeconomic events (such as central bank rate cuts, stock index breaking key levels, etc.)

Criterion 2. Background and Historical Comparison: When explaining, appropriately provide historical interval trends or historical averages to help understand the current position and significance of the indicator
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
DEFAULT_CONCEPT_EXPLANATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CONCEPT_EXPLANATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CONCEPT_EXPLANATION_PROMPT_ZH),
            ),
        ],
    },
)


class ConceptExplanationGrader(LLMGrader):
    """
    Concept Explanation Grader for Finance Domain

    Evaluates the quality of macroeconomic concept explanations by comparing two responses
    based on definition clarity (clear and accurate definitions of indicators, concepts, events)
    and historical context (historical trends and averages for perspective).

    Evaluation Criteria:
    1. Clear and Accurate Definition: Precise definitions of macro indicators, concepts, and events
    2. Background and Historical Comparison: Historical trends and context for understanding

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
        >>> grader = ConceptExplanationGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="什么是实际利率？",
        ...     answer_1="实际利率就是扣除通胀的利率。",
        ...     answer_2="实际利率=名义利率-通胀率。历史上，实际利率均值在2-3%..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_CONCEPT_EXPLANATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize ConceptExplanationGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for concept explanation evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="concept_explanation",
            mode=GraderMode.LISTWISE,
            description="Evaluate macroeconomic concept explanation quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_CONCEPT_EXPLANATION_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two concept explanation responses

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
            logger.error(f"Error evaluating concept explanation: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "concept_explanation",
            "criteria": ["definition_clarity", "historical_context"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
