# -*- coding: utf-8 -*-
"""
Event Identification Grader for Finance Domain

Evaluates the quality of event identification in financial responses by comparing
two responses based on accuracy, time precision, and key element identification.
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
EVENT_IDENTIFICATION_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答对事件识别的准确性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 先判断回答中是否识别到了问题中的事件，能识别到的答案更好;
2. 针对每个评估标准选择更好的回答，并给出你的理由;
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 准确识别：准确检索识别问题中的事件

评价标准2. 时间范围精准：严格匹配问题指定的时间范围，避免模糊表述；

评价标准3. 核心要素准确：是否准确无误地识别和描述了事件的关键信息。例如，在回答"美国CPI数据"问题时，必须准确引用最新的CPI数值、环比与同比增长率、以及核心CPI数据
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
EVENT_IDENTIFICATION_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the accuracy of event identification in financial assistants' responses. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. First, determine whether the event in the question has been identified in the responses. A response that identifies the event is better;
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Accurate Identification: Accurately retrieve and identify the event mentioned in the question

Criterion 2. Time Range Precision: Strictly match the time range specified in the question, avoid vague expressions;

Criterion 3. Core Elements Accuracy: Whether the key information of the event has been accurately identified and described. For example, when answering questions about "US CPI data", must accurately cite the latest CPI value, month-over-month and year-over-year growth rates, and core CPI data
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
DEFAULT_EVENT_IDENTIFICATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(EVENT_IDENTIFICATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(EVENT_IDENTIFICATION_PROMPT_ZH),
            ),
        ],
    },
)


class EventIdentificationGrader(LLMGrader):
    """
    Event Identification Grader for Finance Domain

    Evaluates the quality of event identification in financial responses by comparing
    two responses based on accuracy, time precision, and key element identification.
    This is a pairwise comparison grader that ranks two responses.

    Evaluation Criteria:
    1. Accurate Identification: Whether the event in the question is correctly identified
    2. Time Range Precision: Strictly matching the specified time range
    3. Core Elements Accuracy: Accurate identification and description of key event information

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
        >>> grader = EventIdentificationGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="2024年3月美国CPI数据如何？",
        ...     answer_1="CPI数据有所上涨。",
        ...     answer_2="2024年3月美国CPI同比上涨3.5%，环比上涨0.4%，核心CPI同比上涨3.8%。"
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] means answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_EVENT_IDENTIFICATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize EventIdentificationGrader.

        Args:
            model: The chat model to use for evaluation, either as a BaseChatModel instance or config dict
            template: The prompt template for event identification evaluation.
                     Defaults to DEFAULT_EVENT_IDENTIFICATION_TEMPLATE.
            language: The language for the evaluation prompt. Defaults to LanguageEnum.ZH (Chinese).
        """
        super().__init__(
            name="event_identification",
            mode=GraderMode.LISTWISE,
            description="Evaluate financial event identification quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_EVENT_IDENTIFICATION_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two financial event identification responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better

        Example:
            >>> result = await grader.aevaluate(
            ...     query="2024年3月美国CPI数据如何？",
            ...     answer_1="CPI数据有所上涨。",
            ...     answer_2="2024年3月美国CPI同比上涨3.5%，环比上涨0.4%，核心CPI同比上涨3.8%。"
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
            logger.error(f"Error evaluating event identification: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "event_identification",
            "criteria": ["accurate_identification", "time_precision", "core_elements_accuracy"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
