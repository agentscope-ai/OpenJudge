# -*- coding: utf-8 -*-
"""
Overall Logic Grader for Finance Domain

Evaluates the overall logical structure and coherence of financial analysis responses
by comparing two responses based on clarity, completeness, and consistency.
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
OVERALL_LOGIC_PROMPT_EN = textwrap.dedent(
    """
You are an expert AI Assistant Evaluator. Your task is to evaluate the **Overall Logic Quality** of the AI's response based strictly on the provided evaluation criteria.

<Rubrics>
Please evaluate the response based on **Overall Logic**:
Criterion 1. Whether the Overall Logic is Clear: Assess whether the overall logic of the answer is clear, including but not limited to:
a. Whether it adopts a "summary-analysis-conclusion" format, i.e., stating the conclusion at the beginning, sub-arguments in the middle, and a summary at the end;

Criterion 2. Whether the Overall Logic is Complete: Assess whether the overall logic is complete relative to the financial question, including but not limited to:
a. Whether the sub-arguments are sufficient to derive the conclusion;
b. Whether the overall logic comprehensively answers the financial question;

Criterion 3. Whether the Overall Logic is Consistent: Assess whether the overall logic is self-consistent, including but not limited to:
a. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency;
b. Sub-arguments have no overlap in content, arguing the conclusion from different angles;
c. Sub-arguments have no contradictions or conflicts in content
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
OVERALL_LOGIC_PROMPT_ZH = textwrap.dedent(
    """
你是一个专业的AI助手评估专家。你的任务是根据提供的评估标准，严格评估AI回答的**整体逻辑质量**。

<评分标准>
请根据**整体逻辑**评估回答：
评价标准1. 整体逻辑是否清晰：评估答案整体逻辑是否清晰，包括但不限于：
a. 是否采用"总-分-总"的格式，即开头说出结论，中间是各个子论点，结尾给出总结；

评价标准2. 整体逻辑是否完备：评估答案整体逻辑相对金融问题是否完备，包括但不限于：
a. 子论点是否足够推导出结论；
b. 整体逻辑是否全面回答了金融问题;

评价标准3. 整体逻辑是否自洽：评估答案整体逻辑是否自洽，包括但不限于：
a. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽；
b. 各子论点内容没有重叠，从不同角度论证结论；
c. 各子论点内容没有矛盾或者冲突
</评分标准>

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由；
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
DEFAULT_OVERALL_LOGIC_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=OVERALL_LOGIC_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=OVERALL_LOGIC_PROMPT_ZH,
            ),
        ],
    },
)


class OverallLogicGrader(LLMGrader):
    """
    Overall Logic Grader for Finance Domain

    Purpose:
        Evaluates the overall logical structure and coherence of financial analysis responses
        by comparing two responses based on clarity, completeness, and consistency.

    What it evaluates:
        - Clarity: Clear structure with summary-analysis-conclusion format
        - Completeness: Sufficient sub-arguments, comprehensive answer
        - Consistency: Logical relationships, no overlap, no contradictions

    When to use:
        - Evaluating overall structure of financial analysis
        - Comparing logical coherence of responses
        - Assessing argument organization quality

    Scoring:
        - rank = [1, 2]: Answer 1 is better
        - rank = [2, 1]: Answer 2 is better

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_OVERALL_LOGIC_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.ZH)

    Returns:
        GraderRank object with:
            - rank: [1, 2] or [2, 1] indicating which answer is better
            - reason: Explanation of the evaluation reasoning
            - metadata: Evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from cookbooks.finance_grader.stock_analysis.overall_logic import OverallLogicGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = OverallLogicGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="分析宁德时代的投资价值",
        ...     answer_1="宁德时代值得投资，因为...",
        ...     answer_2="结论：建议买入。理由：1)基本面...2)估值..."
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
        Initialize OverallLogicGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for overall logic evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="overall_logic",
            mode=GraderMode.LISTWISE,
            description="Evaluate overall logic and structure quality by comparing two responses",
            model=model,
            template=template or DEFAULT_OVERALL_LOGIC_TEMPLATE,
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
        Evaluate two responses for overall logic quality

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="分析宁德时代的投资逻辑",
            ...     answer_1="新能源行业前景好。",
            ...     answer_2="投资逻辑：1)行业空间：全球电动车渗透率仅15%..."
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
                    "evaluation_type": "overall_logic",
                    "criteria": ["clarity", "completeness", "consistency"],
                },
            )

        except Exception as e:
            logger.exception(f"Error evaluating overall logic: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["OverallLogicGrader", "DEFAULT_OVERALL_LOGIC_TEMPLATE"]
