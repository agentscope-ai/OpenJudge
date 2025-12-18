# -*- coding: utf-8 -*-
"""Translation: Accurately converts text between languages while preserving meaning,
tone, and cultural context.
"""

import textwrap
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# English Prompts
TRANSLATION_SYSTEM_PROMPT_EN = """You are an expert linguist specializing in translation quality assessment.
Your task is to evaluate the quality of machine translations based on accuracy, fluency, \
and cultural appropriateness."""

TRANSLATION_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are
ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for
your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them
 as supplements.If all responses meet all rubrics, you can judge and choose one answer by yourself.

# Rubrics
Accuracy in Translation:
    Faithfully convey the original text's meaning, intent, and nuances without
    distortion, omission, or addition.
Contextual Appropriateness:
    Preserve the original context, tone, and purpose while adapting to target
    language conventions and specified formatting requirements.
Fluency and Naturalness:
    Produce translations that read naturally in the target language, avoiding
    awkward phrasing or literal translations that compromise readability.
Cultural Sensitivity:
    Appropriately handle culture-specific references, idioms, and concepts,
    ensuring they are correctly adapted or explained for the target audience.

# Query
{query}

# Responses
{responses}

# Output Requirement
```json
{{
    "rank": ["The rank score of the answer in the list."],
    "reason": "The reason for the score."
}}
```
"""

# Chinese Prompts
TRANSLATION_SYSTEM_PROMPT_ZH = (
    """您是一位专注于翻译质量评估的专家语言学家。您的任务是根据准确性、流畅性和文化适宜性来评估机器翻译的质量。"""
)

TRANSLATION_USER_PROMPT_ZH = """# 任务描述
您的角色是专业评估专家。我将为您提供一个问题和几个候选答案。您的任务是从候选答案中选出最佳的一个。
我还会为您提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。这些评分标准可以作为您判断的补充知识。
如果您认为任何评分标准对当前问题有帮助，请随意使用它们作为补充。如果所有答案都满足所有评分标准，您可以自行判断并选择一个答案。

# 评分标准
翻译准确性：
    忠实地传达原文的含义、意图和细微差别，不扭曲、省略或添加内容。
语境适宜性：
    保留原始语境、语气和目的，同时适应目标语言惯例和指定的格式要求。
流畅性和自然性：
    产生在目标语言中读起来自然的翻译，避免影响可读性的别扭措辞或字面翻译。
文化敏感性：
    适当地处理特定文化参照、习语和概念，确保它们为目标受众正确调整或解释。

# 查询
{query}

# 答案
{responses}

# 输出要求
```json
{{
    "rank": ["答案在列表中的排名分数。"],
    "reason": "得分原因。"
}}
```
"""

TRANSLATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=textwrap.dedent(TRANSLATION_SYSTEM_PROMPT_EN).strip(),
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRANSLATION_USER_PROMPT_EN).strip(),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=textwrap.dedent(TRANSLATION_SYSTEM_PROMPT_ZH).strip(),
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRANSLATION_USER_PROMPT_ZH).strip(),
            ),
        ],
    },
)


class TranslationGrader(LLMGrader):
    """TranslationGrader

    Accurately converts text between languages while preserving meaning, tone,
    and cultural context.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = TRANSLATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the TranslationGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to LISTWISE.
            language: The language of the prompts. Defaults to English.
            structured_model: The Pydantic model for structured output parsing.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="translation",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template or TRANSLATION_TEMPLATE,
            language=language,
            description="Accurately converts text between languages while preserving meaning, "
            "tone, and cultural context.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the translation based on the query.

        Evaluates translation responses for their ability to accurately convert
        text between languages while preserving meaning, tone, and cultural context.
        The grader focuses on accuracy, fluency, and cultural appropriateness.

        Args:
            query (str): The original text to be translated.
            answer (List[str]): The translations to evaluate.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: Contains a ranked list and explanation.
                - rank (List[int]): Ranking of translations by quality
                - reason (str): Explanation of how the ranking was determined
                - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for listwise translation grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = TranslationGrader(mode=GraderMode.LISTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Hello, how are you today?",
            ...     responses=["Hola, ¿cómo estás hoy?", "¡Hola! ¿Qué tal?"]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first translation is more accurate and preserves the formal tone.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
