# -*- coding: utf-8 -*-
"""Role Playing: Engages in immersive roleplay scenarios with consistent
character portrayal and contextual awareness.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderRank
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Role Playing Listwise System Prompt
ROLE_PLAYING_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

ROLE_PLAYING_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Role Playing Listwise User Prompt
ROLE_PLAYING_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics \
are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge \
for your judgment, though not necessarily required. First, think independently. Use these rubrics \
only when unsure about certain responses, selecting specific ones based on the questions and responses.

# Rubrics
Character and Contextual Fidelity:
    Prioritize maintaining the assigned character's persona, motivations, and
    world-building consistency while strictly adhering to the scenario's
    established rules, terminology, and thematic boundaries to ensure immersive
    authenticity.
Consistent Character Voice:
    Ensure the character's speech patterns, vocabulary, and mannerisms are
    consistent throughout the interaction, reflecting their background,
    personality, and current emotional state.
Contextual Appropriateness:
    Responses should be appropriate to the scenario's setting, time period, and
    cultural context, avoiding anachronisms or out-of-place references that
    break immersion.
Engagement Quality:
    The interaction should be engaging and maintain the role-playing scenario's
    momentum, with the character responding naturally to prompts while adding
    depth to the narrative.

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

ROLE_PLAYING_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还将为你提供一组评分标准，列在标题#评分标准下。这些评分标准按重要性从高到低排列。这些评分标准可以作为你判断的补充知识，
尽管不一定必需。首先，独立思考。只有在对某些答案不确定时才使用这些评分标准，根据问题和答案选择特定的答案。

# 评分标准
角色和情境保真度：
    优先保持分配的角色的人物形象、动机和世界观的一致性，同时严格遵守情境设定的既定规则、术语和主题边界，以确保沉浸式的
    真实感。
一致的角色声音：
    确保角色的说话模式、词汇和举止在整个互动过程中保持一致，反映出他们的背景、个性和当前情绪状态。
情境适宜性：
    回答应该适合情境的设定、时代和文化背景，避免破坏沉浸感的时代错误或不合时宜的引用。
参与质量：
    互动应该是引人入胜的，并保持角色扮演情境的势头，角色自然地回应提示，同时为叙事增添深度。

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

ROLE_PLAYING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=ROLE_PLAYING_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=ROLE_PLAYING_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=ROLE_PLAYING_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=ROLE_PLAYING_USER_PROMPT_ZH),
        ],
    },
)


class RolePlayingGrader(LLMGrader):
    """Role Playing

    Engages in immersive roleplay scenarios with consistent character
    portrayal and contextual awareness.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = ROLE_PLAYING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the RolePlayingGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="role_playing",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Engages in immersive roleplay scenarios with consistent character "
            "portrayal and contextual awareness.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the role playing response based on the query.

        Evaluates role playing responses for their ability to engage in immersive
        roleplay scenarios with consistent character portrayal and contextual awareness.
        The grader focuses on character consistency, contextual awareness, and
        immersive engagement.

        Args:
            query (str): The role playing scenario or prompt.
            responses (List[str]): The role playing responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> # Example for listwise role playing grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = RolePlayingGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="You are a medieval blacksmith. A knight approaches requesting a sword.",
            ...     responses=[
            ...         "Ah, good sir knight! I shall forge you a blade worthy of your noble quest. "
            ...         "What specifications would you desire in your weapon?",
            ...         "Hello, how can I help you today?"
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] 第一个回答更好地维持了角色一致性和情境真实性。
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
