# -*- coding: utf-8 -*-
"""FocusGrader

Maintains strict adherence to the main topic while filtering out irrelevant information.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Focus Listwise System Prompt
FOCUS_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

FOCUS_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Focus Listwise User Prompt
FOCUS_USER_PROMPT_EN = """# Task Description
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
Direct Relevance to Core Query:
    Prioritize completions that explicitly address the specific question, task, \
or scenario posed in the query without introducing tangential concepts, \
unnecessary details, or unrelated analysis.
Maintaining Central Theme:
    Ensure the response stays focused on the main topic throughout, avoiding \
digressions or shifts to peripheral subjects.
Filtering Irrelevant Information:
    Eliminate or avoid including information that does not directly contribute \
to answering the core query or completing the primary task.
Adhering to Length Constraints:
    Provide responses that are appropriately detailed without unnecessary \
elaboration, keeping the focus sharp and concise.

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

FOCUS_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还将为你提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。你必须依次检查每个候选答案，看它是否违反任何评分标准，并提供你发现的任何违规原因。这些原因应该被用作对答案进行排名的参考。
你可以按自己认为合适的方式组织推理，但要尽量保持思维过程简洁。

# 评分标准
与核心查询的直接相关性：
    优先考虑明确解决查询中提出的特定问题、任务或场景的完成内容，不引入切线概念、不必要的细节或无关的分析。
保持中心主题：
    确保响应始终专注于主要话题，避免离题或转向外围主题。
过滤无关信息：
    消除或避免包含不直接有助于回答核心查询或完成主要任务的信息。
遵守长度限制：
    提供适当详细的响应，而无需不必要的阐述，保持焦点清晰简洁。

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

FOCUS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=FOCUS_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=FOCUS_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=FOCUS_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=FOCUS_USER_PROMPT_ZH),
        ],
    },
)


class FocusGrader(LLMGrader):
    """FocusGrader

    Maintains strict adherence to the main topic while filtering out irrelevant information.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = FOCUS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the FocusGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="focus",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Maintains strict adherence to the main topic while filtering out " "irrelevant information.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> Any:
        """Evaluate the focus quality of the response based on the query.

        Evaluates responses for their ability to maintain strict adherence to the
        main topic while filtering out irrelevant information. The grader focuses
        on direct relevance to the core query and elimination of tangential content.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The list of responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> grader = FocusGrader()
            >>> result = await grader.aevaluate(
            ...     query="Explain the process of photosynthesis",
            ...     responses=[
            ...         "Photosynthesis is the process by which plants convert light "
            ...            "energy into chemical energy...",
            ...         "Plants are green and need water to survive..."
            ...     ]
            ... )
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
