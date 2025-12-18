# -*- coding: utf-8 -*-
"""Closed QA Grader

Provides precise, fact-based responses to questions with definitive correct responses.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Closed QA Listwise System Prompt (English)
CLOSED_QA_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

# Closed QA Listwise System Prompt (Chinese)
CLOSED_QA_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Closed QA Listwise User Prompt (English)
CLOSED_QA_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Factual Accuracy:
    Prioritize completely accurate information without any factual errors or
    hallucinations. Every statement should be verifiable against authoritative
    sources.
Precision and Conciseness:
    Provide responses that directly and precisely answer the question without
    unnecessary elaboration or ambiguity.
Comprehensiveness within Scope:
    Include all relevant information required to fully answer the question, but
    avoid including tangential or excessive details.
Logical Coherence:
    Structure responses in a clear, logical manner that enhances understanding
    and maintains focus on the core question.

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

# Closed QA Listwise User Prompt (Chinese)
CLOSED_QA_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
事实准确性：
    优先考虑完全准确的信息，不能有任何事实错误或幻觉。每个陈述都应该能够根据权威来源进行验证。
精确性和简洁性：
    提供直接且精确地回答问题的响应，无需不必要的阐述或歧义。
范围内的全面性：
    包含充分回答问题所需的所有相关信息，但避免包含无关或过多的细节。
逻辑连贯性：
    以清晰、逻辑的方式构建响应，增强理解并保持对核心问题的关注。

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

CLOSED_QA_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=CLOSED_QA_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=CLOSED_QA_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=CLOSED_QA_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=CLOSED_QA_USER_PROMPT_ZH,
            ),
        ],
    },
)


class ClosedQAGrader(LLMGrader):
    """Closed QA

    Provides precise, fact-based responses to questions with definitive correct responses.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = CLOSED_QA_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the ClosedQAGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="closed_qa",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Provides precise, fact-based responses to questions with definitive " "correct responses.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the closed QA response based on the query.

        Evaluates closed QA responses for their ability to provide precise,
        fact-based responses to questions with definitive correct responses.
        The grader emphasizes precision in fact-based responses, avoids
        hallucinations, and focuses on explicit requirements.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The list of responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> # Example for listwise closed QA grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = ClosedQAGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     responses=[
            ...         "The capital of France is Paris.",
            ...         "The capital of France is Lyon."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first answer correctly identifies Paris as the capital of France.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
