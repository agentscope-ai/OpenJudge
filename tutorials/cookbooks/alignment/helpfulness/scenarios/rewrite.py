# -*- coding: utf-8 -*-
"""RewriteGrader."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Rewrite Listwise System Prompt
REWRITE_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

REWRITE_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Rewrite Listwise User Prompt (English)
REWRITE_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with an original text \
and several rewritten versions. Your task is to select the single best rewritten version from the \
candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics \
are ordered from highest to lowest importance. You must check each rewritten version in turn to \
see if it violates any rubric, and provide reasons for any violations you find. These reasons \
should be used as references for ranking the rewritten versions.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

# Rubrics
Preservation of Meaning:
    The rewritten text must accurately preserve the original meaning without introducing factual errors or distortions.
Enhancement of Clarity:
    The rewritten text should be clearer and easier to understand than the original, with improved logical flow.
Style and Format Adaptation:
    The rewritten text should appropriately adapt its style and format to better suit the target audience or purpose.
Elimination of Redundancy:
    Remove unnecessary repetition while retaining essential information.
Improved Readability:
    Enhance sentence structure, vocabulary, and overall readability appropriate to the target audience.

# Original Text
{query}

# Rewritten Versions
{responses}

# Output Requirement
```json
{{
    "rank": ["The rank score of the rewritten versions in the list."]
    "reason": "The reason for the score."
}}
```
"""

# Rewrite Listwise User Prompt (Chinese)
REWRITE_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供原文和几个重写版本。你的任务是从候选版本中选择最佳的重写版本。
我还会为你提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。你必须依次检查每个重写版本，
看它是否违反了任何评分标准，并提供你发现的任何违规原因。这些原因应该作为对重写版本进行排名的参考。
你可以按自己认为合适的方式组织推理过程，但要保持思维过程尽可能简洁。

# 评分标准
保留原意：
    重写文本必须准确保留原意，不得引入事实错误或扭曲。
提升清晰度：
    重写文本应该比原文更清晰易懂，逻辑流程得到改善。
风格和格式调整：
    重写文本应适当调整其风格和格式，以更好地适合目标受众或目的。
消除冗余：
    删除不必要的重复，同时保留基本信息。
提高可读性：
    提高句子结构、词汇和整体可读性，使之适合目标受众。

# 原文
{query}

# 重写版本
{answer}

# 输出要求
```json
{{
    "rank": ["重写版本在列表中的排名分数。"]
    "reason": "得分的原因。"
}}
```
"""

REWRITE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=REWRITE_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=REWRITE_USER_PROMPT_ZH),
        ],
    },
)


class RewriteGrader(LLMGrader):
    """Rewrite Grader

    Improves text quality by correcting errors and enhancing clarity,
    fluency, and style.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = REWRITE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the RewriteGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template
                      will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="rewrite",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Improves text quality by correcting errors and enhancing clarity, " "fluency, and style.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the rewritten text based on the query.

        Evaluates rewrite responses for their ability to improve text quality by
        correcting errors and enhancing clarity, fluency, and style. The grader
        focuses on error correction, clarity improvement, and stylistic enhancement.

        Args:
            query (str): The original text to be rewritten.
            responses (List[str]): The rewritten texts to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: Contains a ranked list and explanation.
                - rank (List[int]): Ranking of rewritten texts by quality
                - reason (str): Explanation of how the ranking was determined
                - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for listwise rewrite grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = RewriteGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="This is a bad writen sentance with alot of erors.",
            ...     responses=[
            ...         "This is a poorly written sentence with many errors.",
            ...         "This sentence has grammatical mistakes that need fixing."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
        """
        # Format responses as numbered list
        formatted_responses = "\n".join([f"{i}. {ans}" for i, ans in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, answer=formatted_responses, **kwargs)
