# -*- coding: utf-8 -*-
"""Precise-IF

Provides concise, targeted responses that directly address conditional queries.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Precise If Listwise System Prompt
PRECISE_IF_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

PRECISE_IF_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Precise If Listwise User Prompt
PRECISE_IF_USER_PROMPT_EN = """# Task Description
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
Strict Adherence to Explicit Formatting and Structural Requirements:
    Prioritize exact compliance with all specified formatting, structural, and technical constraints (e.g., punctuation, indentation, bullet points, word counts) as the primary criterion for evaluating completions.
Clarity, Logical Progression, and Thematic Consistency:
    Ensure content is coherent, logically structured, and maintains alignment with the scenario's core premise, fulfilling implicit demands for depth, relevance, and narrative or analytical consistency.
Conditional Accuracy and Contextual Appropriateness:
    Verify that responses correctly interpret and apply conditional logic, providing accurate information that is appropriate for the given context and conditions.
Explicit Condition Matching:
    Confirm that responses directly address specified conditions and constraints, without omitting critical elements or introducing irrelevant information.

# Query
{query}

# Responses
{answer}

# Output Requirement
```json
{{
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}}
```
"""

PRECISE_IF_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还将为你提供一组评分标准，列在标题#评分标准下。这些评分标准按重要性从高到低排列。你必须逐一检查每个候选答案是否违反任何评分标准，并提供违反原因。这些原因应该作为排名答案的参考。
你可以按自己的想法组织推理过程，但要尽量保持思维过程简洁。

# 评分标准
严格遵守明确的格式和结构要求：
    将完全符合所有指定的格式、结构和技术约束（例如标点符号、缩进、项目符号、字数）作为评估完成内容的主要标准。
清晰度、逻辑进展和主题一致性：
    确保内容连贯、逻辑结构清晰，并与场景的核心前提保持一致，满足深度、相关性和叙事或分析一致性的隐含要求。
条件准确性和情境适当性：
    验证响应是否正确解释和应用条件逻辑，提供适合给定情境和条件的准确信息。
明确条件匹配：
    确认响应直接解决了指定的条件和约束，没有省略关键要素或引入无关信息。

# 查询
{query}

# 回答
{answer}

# 输出要求
```json
{{
    "rank": ["答案在列表中的排名分数。"]
    "reason": "得分的原因。"
}}
```
"""

PRECISE_IF_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=PRECISE_IF_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=PRECISE_IF_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=PRECISE_IF_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=PRECISE_IF_USER_PROMPT_ZH,
            ),
        ],
    },
)


class PreciseIfGrader(LLMGrader):
    """Precise-IF

    Provides concise, targeted responses that directly address conditional queries.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = PRECISE_IF_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the PreciseIfGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="precise_if",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Provides concise, targeted responses that directly address " "conditional queries.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the precision of the IF response based on the query.

        Evaluates IF responses for their ability to provide concise, targeted responses
        that directly address conditional queries. The grader focuses on precision,
        relevance, and direct adherence to specified conditions.

        Args:
            query (str): The conditional query to evaluate.
            responses (List[str]): The responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> # Example for listwise precise IF grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = PreciseIfGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="If the temperature is above 30°C, what should I wear?",
            ...     responses=[
            ...         "If the temperature is above 30°C, wear light, breathable clothing such as shorts "
            ...         "and a t-shirt.",
            ...         "Wear whatever you want."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first response directly addresses the conditional query with specific, appropriate
                suggestions.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
