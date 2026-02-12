# -*- coding: utf-8 -*-
"""
Valuation Analysis Grader for Finance Domain

Evaluates the quality of valuation analysis by comparing two responses based on
conclusion clarity, completeness, and logical rigor.
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
VALUATION_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Valuation Analysis Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Valuation Analysis**:
Criterion 1. Clear Valuation Conclusion: The response must have a valuation conclusion, such as overvalued/undervalued, etc.

Criterion 2. Valuation Completeness: The response includes multi-dimensional valuation analysis, including but not limited to:
a. Current valuation compared to historical valuation
b. Comparison with peer companies in the same industry
c. Comparison with industry valuation benchmark

Criterion 3. Logical Rigor of Valuation: The valuation conclusion has a complete logical chain, cannot have only conclusions without evidence
</Rubrics>

<Steps>
1. First, determine whether there is valuation analysis in the responses. A response with valuation analysis is better than one without; if both responses lack valuation analysis, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Steps>

<Constraints>
- **Objectivity**: Do not be influenced by the politeness or length of the response.
- **Focus**: Only evaluate based on the provided evaluation criteria.
- A response with valuation analysis is better than one without.
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
VALUATION_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**估值分析质量**。

<评分标准>
请根据**估值分析**评估回答：
评价标准1. 估值结论清晰：评估回答中必须有估值结论，例如偏高/偏低等

评价标准2. 估值完整性：评估回答中包含多个维度的估值分析，包含但不限于：
a. 当前估值与历史估值对比
b. 与同行业可比公司对比
c. 与行业估值中枢对比

评价标准3. 估值逻辑严谨性：评估回答中估值结论有完整的逻辑链，不能只有结论没有论据
</评分标准>

<评估步骤>
1. 先判断回答中有无估值分析，有估值分析的回答优于没有估值分析的回答；如果两个回答都没有估值分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由；
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<注意事项>
- **客观性**：不要被回答的礼貌程度或长度所影响。
- **聚焦**：仅根据提供的评估标准进行评估。
- 有估值分析的回答优于没有估值分析的回答。
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
DEFAULT_VALUATION_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=VALUATION_ANALYSIS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=VALUATION_ANALYSIS_PROMPT_ZH,
            ),
        ],
    },
)


class ValuationAnalysisGrader(LLMGrader):
    """
    Valuation Analysis Grader for Finance Domain

    Purpose:
        Evaluates the quality of valuation analysis by comparing two responses based on
        conclusion clarity, completeness, and logical rigor.

    What it evaluates:
        - Clear Conclusion: Must have clear valuation conclusion (overvalued/undervalued)
        - Completeness: Multi-dimensional comparisons (historical, peers, industry)
        - Logical Rigor: Complete logical chain with evidence

    When to use:
        - Evaluating stock valuation analysis responses
        - Comparing quality of valuation assessment
        - Assessing depth of valuation methodology

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_VALUATION_ANALYSIS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_analysis.valuation_analysis import ValuationAnalysisGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = ValuationAnalysisGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="宁德时代估值是否合理？",
        ...     answer_1="估值偏高。",
        ...     answer_2="结论：估值偏高。PE 60倍，高于历史均值45倍，也高于比亚迪的50倍..."
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
        Initialize ValuationAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for valuation analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="valuation_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate valuation analysis quality by comparing two responses",
            model=model,
            template=template or DEFAULT_VALUATION_ANALYSIS_TEMPLATE,
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
        Evaluate two valuation analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析宁德时代的估值水平",
            ...     answer_1="估值较高。",
            ...     answer_2="当前PE 35倍，PEG 1.2，相对行业平均PE 25倍有溢价..."
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
                    "evaluation_type": "valuation_analysis",
                    "criteria": ["conclusion_clarity", "completeness", "logical_rigor"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating valuation analysis: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["ValuationAnalysisGrader", "DEFAULT_VALUATION_ANALYSIS_TEMPLATE"]
