# -*- coding: utf-8 -*-
"""SummarizationGrader."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Summarization Listwise System Prompt
SUMMARIZATION_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

SUMMARIZATION_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Summarization Listwise User Prompt
SUMMARIZATION_USER_PROMPT_EN = """# Task Description

Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Comprehensive Coverage of Core Content: A superior summary captures all critical \
elements, themes, and details central to the source material without omitting key \
information. Avoidance of Irrelevant or Tangential Information: Focuses exclusively \
on the primary subject, eliminating extraneous details that distract from the core \
narrative or argument. Logical Structure and Coherence: Information is organized in \
a clear, hierarchical, or chronological sequence to ensure readability and logical \
progression of ideas. Factual Accuracy and Neutral Objectivity: The summary must \
faithfully represent the source material without introducing distortions, opinions, \
or subjective interpretations, maintaining a neutral tone throughout.

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

SUMMARIZATION_USER_PROMPT_ZH = """# 任务描述

你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
全面覆盖核心内容：优秀的摘要应捕捉源材料中的所有关键元素、主题和细节，不遗漏重要信息。\
避免无关或离题信息：专注于主要主题，消除分散核心叙事或论点注意力的无关细节。\
逻辑结构和连贯性：信息应以清晰的层次或时间顺序组织，确保可读性和思想的逻辑推进。\
事实准确性和中立客观性：摘要必须忠实地代表源材料，不引入扭曲、观点或主观解释，始终保持中立语气。

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

SUMMARIZATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=SUMMARIZATION_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=SUMMARIZATION_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=SUMMARIZATION_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=SUMMARIZATION_USER_PROMPT_ZH),
        ],
    },
)


class SummarizationGrader(LLMGrader):
    """Summarization: Condenses information while preserving key points and overall meaning."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = SUMMARIZATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the SummarizationGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="summarization",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Condenses information while preserving key points and overall meaning.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the summary based on the query.

        Evaluates summarization responses for their ability to condense information
        while preserving key points and overall meaning. The grader focuses on
        key point preservation, conciseness, and clarity.

        Args:
            query (str): The original text to be summarized.
            responses (List[str]): The list of summaries to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> import asyncio
            >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = SummarizationGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Climate change is a significant global challenge that affects "
                          "ecosystems, economies, and societies worldwide. It is caused primarily "
                          "by human activities such as burning fossil fuels, deforestation, and "
                          "industrial processes. The impacts include rising sea levels, extreme "
                          "weather events, biodiversity loss, and food security issues. Addressing "
                          "climate change requires international cooperation, policy changes, and "
                          "technological innovations.",
            ...     responses=[
            ...         "Climate change, caused by human activities, has global impacts on "
                           "ecosystems and societies. It requires international cooperation and "
                           "technological solutions.",
            ...         "Global warming affects our planet. We should do something about it."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
