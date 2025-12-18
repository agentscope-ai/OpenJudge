# -*- coding: utf-8 -*-
"""Brainstorming: Generates creative ideas and suggestions to address user challenges."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Brainstorming Listwise System Prompt
BRAINSTORMING_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

BRAINSTORMING_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Brainstorming Listwise User Prompt
BRAINSTORMING_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Creative Relevance and Contextual Alignment:
    Prioritize completions that balance novel ideas with direct ties to the scenario's core context, ensuring ideas are both imaginative and grounded in the specific problem or theme.
Practical Feasibility and Actionable Detail:
    Favor completions that offer concrete, implementable solutions or insights, avoiding abstract or overly speculative suggestions that lack real-world applicability.
Structural Coherence and Logical Organization:
    Prefer completions that present ideas in a clear, logically sequenced framework (e.g., categorized sections, step-by-step processes) to enhance readability and development potential.

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

BRAINSTORMING_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
创造性相关性和情境对齐：
    优先考虑那些在新颖想法和情境核心内容之间取得平衡的完成内容，确保想法既有想象力又立足于特定问题或主题。
实用可行性和可操作细节：
    偏好提供具体、可实施的解决方案或见解的完成内容，避免缺乏现实适用性的抽象或过于投机的建议。
结构连贯性和逻辑组织：
    偏好以清晰、逻辑顺序呈现想法的完成内容（例如，分类章节、逐步过程），以增强可读性和发展潜力。

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

BRAINSTORMING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=BRAINSTORMING_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=BRAINSTORMING_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=BRAINSTORMING_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=BRAINSTORMING_USER_PROMPT_ZH),
        ],
    },
)


class BrainstormingGrader(LLMGrader):
    """Brainstorming: Generates creative ideas and suggestions to address user challenges."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = BRAINSTORMING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the BrainstormingGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="brainstorming",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Generates creative ideas and suggestions to address user challenges.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the brainstorming response based on the query.

        Evaluates brainstorming responses for their ability to generate creative
        ideas and suggestions that address user challenges. The grader focuses on
        idea diversity, relevance to the core question, and practical implementation
        guidance.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The list of responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> grader = BrainstormingGrader()
            >>> result = await grader.aevaluate(
            ...     query="Give me ideas for a birthday gift for my 10-year-old",
            ...     responses=[
            ...         "Here are some ideas: 1) Art supplies kit, 2) Science experiment set,"
            ...         " 3) Board game, 4) Book series",
            ...         "Consider these options: toys, games, books, or educational kits"
            ...     ]
            ... )
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
