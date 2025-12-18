# -*- coding: utf-8 -*-
"""ReasoningGrader."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Reasoning Listwise System Prompt
REASONING_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

REASONING_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Reasoning Listwise User Prompt
REASONING_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Logical Soundness:
    Ensure all reasoning steps follow logically from premises and established
    rules of inference, avoiding logical fallacies or invalid conclusions.
Step-by-Step Clarity:
    Present the reasoning process in a clear, sequential manner that allows for
    easy verification and understanding of how conclusions are reached.
Evidence-Based Conclusions:
    Base conclusions on solid evidence, clearly distinguishing between facts,
    assumptions, and inferences.
Comprehensiveness:
    Address all aspects of the problem or question, considering alternative
    approaches and potential edge cases where relevant.

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

REASONING_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
逻辑健全性：
    确保所有推理步骤都遵循前提和既定的推理规则，避免逻辑谬误或无效结论。
逐步清晰性：
    以清晰、有序的方式呈现推理过程，便于验证和理解结论是如何得出的。
基于证据的结论：
    结论应基于确凿的证据，清楚地区分事实、假设和推论。
全面性：
    解决问题或回答问题的所有方面，考虑替代方法和相关边缘情况。

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

REASONING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=REASONING_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=REASONING_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=REASONING_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=REASONING_USER_PROMPT_ZH,
            ),
        ],
    },
)


class ReasoningGrader(LLMGrader):
    """ReasoningGrader

    Applies logical thinking and systematic approaches to solve problems and answer questions.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = REASONING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the ReasoningGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="reasoning",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Applies logical thinking and systematic approaches to solve problems " "and answer questions.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the reasoning quality of the response based on the query.

        Evaluates reasoning responses for their ability to apply logical thinking
        and systematic approaches to solve problems and answer questions. The
        grader focuses on logical soundness, step-by-step clarity, and
        evidence-based conclusions.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The list of responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> # Example for listwise reasoning grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = ReasoningGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="If all roses are flowers and some flowers are red, is it true that "
            ...           "some roses are red?",
            ...     responses=[
            ...         "This is undetermined because we don't know which flowers are red.",
            ...         "Yes, since roses are flowers and some flowers are red, some roses must be red."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first answer demonstrates better logical reasoning by correctly identifying
                  the undetermined nature of the statement.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
