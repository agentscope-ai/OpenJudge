# -*- coding: utf-8 -*-
"""Open QA Grader

Provides comprehensive, nuanced responses to open-ended questions without definitive correct
responses.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Open QA Rank System Prompt
OPEN_QA_SYSTEM_PROMPT_EN = """You are a helpful assistant skilled in reward evaluation.
Please make reward judgments based on the given prompt words."""

OPEN_QA_SYSTEM_PROMPT_ZH = """你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"""

# Open QA Rank User Prompt
OPEN_QA_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. \
These rubrics are ordered from highest to lowest importance. You must check each \
candidate answer in turn to see if it violates any rubric, and provide reasons for \
any violations you find. These reasons should be used as references for ranking \
the responses.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

# Rubrics
Comprehensive Coverage:
    Address all aspects of the question thoroughly, providing a well-rounded \
response that considers multiple angles and relevant factors without \
omitting significant points.
Source Reliability and Citation:
    Base information on credible, up-to-date sources and clearly attribute \
facts or claims to their origins when appropriate, particularly for \
contentious or specialized topics.
Nuanced Understanding:
    Demonstrate sophisticated comprehension of the topic by acknowledging \
complexities, uncertainties, or trade-offs rather than presenting overly \
simplified or one-sided views.
Clarity and Organization:
    Present information in a clear, logically structured format that enhances \
readability and facilitates understanding, using appropriate headings, \
paragraphs, and transitions.

# Query
{query}

# Responses
{responses}

# Output Requirement
```json
{{
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}}
```
"""

OPEN_QA_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还将为你提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。你必须依次检查每个候选答案，看它是否违反任何评分标准，并提供违反原因。这些原因应作为排名的答案的参考。
你可以按自己认为合适的方式组织推理，但要尽量保持思维过程简洁。

# 评分标准
全面覆盖：
    全面深入地解答问题，提供一个考虑多个角度和相关因素的均衡回应，不遗漏重要要点。
来源可靠性和引用：
    基于可信、最新的信息来源，并在适当时清楚地标明事实或主张的出处，特别是对于有争议或专业性强的话题。
细致理解：
    通过对主题复杂性、不确定性或权衡的承认，展示对该主题的深度理解，而不是呈现过度简化或片面的观点。
清晰度和组织性：
    以清晰、逻辑结构化的格式呈现信息，增强可读性并促进理解，使用适当的标题、段落和过渡。

# 查询
{query}

# 回答
{responses}

# 输出要求
```json
{{
    "rank": ["答案在列表中的排名分数。"]
    "reason": "得分的原因。"
}}
```
"""

OPEN_QA_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=OPEN_QA_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=OPEN_QA_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=OPEN_QA_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=OPEN_QA_USER_PROMPT_ZH,
            ),
        ],
    },
)


class OpenQAGrader(LLMGrader):
    """OpenQAGrader

    Provides comprehensive, nuanced responses to open-ended questions without definitive correct
    responses.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = OPEN_QA_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the OpenQAGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="open_qa",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Provides comprehensive, nuanced responses to open-ended questions without "
            "definitive correct responses.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the open QA response based on the query.

        Evaluates open QA responses for their comprehensiveness, source reliability,
        nuanced understanding, and clarity. The grader focuses on addressing all
        aspects of the question and providing well-rounded responses.

        Args:
            query (str): The open-ended question to evaluate.
            responses (List[str]): The responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: Contains a ranked list and explanation.
                - rank (List[int]): Ranking of responses by quality
                - reason (str): Explanation of how the ranking was determined
                - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise open QA grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = OpenQAGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="What are the potential impacts of climate change on global agriculture?",
            ...     responses=[
            ...         "Climate change can impact global agriculture in several ways...",
            ...         "Global warming affects farming practices worldwide..."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first answer is more comprehensive and better organized.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
