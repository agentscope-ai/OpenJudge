# -*- coding: utf-8 -*-
"""
Concept Explanation Grader for Finance Domain

Evaluates the quality of macroeconomic concept explanations by comparing two responses
based on definition clarity and historical context.
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
CONCEPT_EXPLANATION_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Macroeconomic Concept Explanation Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Concept Explanation**:
Criterion 1. Clear and Accurate Definition: Provide clear and accurate definitions for macroeconomic indicators (such as interest rates, CPI, GDP, retail sales, PMI, etc.), macroeconomic concepts (such as real interest rates, potential nominal growth rate, yield curve inversion, etc.), and macroeconomic events (such as central bank rate cuts, stock index breaking key levels, etc.)

Criterion 2. Background and Historical Comparison: When explaining, appropriately provide historical interval trends or historical averages to help understand the current position and significance of the indicator
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
CONCEPT_EXPLANATION_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**宏观概念解释质量**。

<评分标准>
请根据**概念解释**评估回答：
评价标准1. 定义清晰准确：对宏观指标（如利率、CPI、GDP、社零、PMI等）、宏观概念（如实际利率、潜在名义增速、收益率倒挂等）、宏观事件（如央行降息、股指突破关键点位等）给出清晰准确的的定义

评价标准2. 背景与历史对比：在解释时适当给出历史区间的变化趋势或历史均值，帮助理解指标当前所处的位置和意义
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
DEFAULT_CONCEPT_EXPLANATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=CONCEPT_EXPLANATION_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=CONCEPT_EXPLANATION_PROMPT_ZH,
            ),
        ],
    },
)


class ConceptExplanationGrader(LLMGrader):
    """
    Concept Explanation Grader for Finance Domain

    Purpose:
        Evaluates the quality of macroeconomic concept explanations by comparing two responses
        based on definition clarity and historical context.

    What it evaluates:
        - Clear and Accurate Definition: Precise definitions of macro indicators, concepts, and events
        - Background and Historical Comparison: Historical trends and context for understanding

    When to use:
        - Evaluating macroeconomic concept explanation responses
        - Comparing quality of financial term definitions
        - Assessing depth of historical context

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_CONCEPT_EXPLANATION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.macro_analysis.concept_explanation import ConceptExplanationGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = ConceptExplanationGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="什么是实际利率？",
        ...     answer_1="实际利率就是扣除通胀的利率。",
        ...     answer_2="实际利率=名义利率-通胀率。历史上，实际利率均值在2-3%..."
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
            template=template or DEFAULT_CONCEPT_EXPLANATION_TEMPLATE,
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
        Evaluate two concept explanation responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="什么是量化宽松政策？",
            ...     answer_1="量化宽松是一种货币政策。",
            ...     answer_2="量化宽松(QE)是央行通过购买国债等资产向市场注入流动性..."
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
                    "evaluation_type": "concept_explanation",
                    "criteria": ["definition_clarity", "historical_context"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating concept explanation: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["ConceptExplanationGrader", "DEFAULT_CONCEPT_EXPLANATION_TEMPLATE"]
