# -*- coding: utf-8 -*-
"""GenerationGrader

Creates high-quality, instruction-following content across diverse formats and topics.
"""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Generation Rank System Prompt
GENERATION_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

GENERATION_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Generation Rank User Prompt
GENERATION_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.

# Rubrics
Adherence to Instructional Specificity:
    Prioritize addressing all explicit requirements (e.g., format, content scope,
    tone) with precise alignment to ensure completeness and fidelity to the
    task's intent.
Depth and Originality in Content:
    Deliver nuanced, actionable insights or creative elements that exceed generic
    responses through specific examples, contextual relevance, and imaginative
    elaboration.
Structural Coherence and Logical Flow:
    Maintain organized progression (e.g., clear hierarchy, thematic sequencing)
    to enhance readability while avoiding contradictions or deviations from
    established frameworks.

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

GENERATION_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还会为你提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。这些评分标准可以作为你判断的补充知识。如果你发现任何评分标准对当前问题有帮助，请随意将其作为补充。

# 评分标准
遵循指令特异性：
    优先考虑解决所有明确要求（例如，格式、内容范围、语气），
    精确对齐以确保完整性和忠实于任务意图。
内容深度和原创性：
    通过具体示例、情境相关性和富有想象力的阐述，
    提供超越一般回应的细致入微、可操作的见解或创意元素。
结构连贯性和逻辑流程：
    保持有序进展（例如，清晰的层次结构、主题序列），
    以增强可读性，同时避免与既定框架的矛盾或偏离。

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

GENERATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=GENERATION_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=GENERATION_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=GENERATION_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=GENERATION_USER_PROMPT_ZH,
            ),
        ],
    },
)


class GenerationGrader(LLMGrader):
    """GenerationGrader

    Creates high-quality, instruction-following content across diverse formats and topics.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = GENERATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the GenerationGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to LISTWISE.
            rubrics: Custom rubrics for evaluation. If None, default rubrics will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="generation",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Creates high-quality, instruction-following content across diverse " "formats and topics.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the response content based on the query.

        Evaluates content generation for adherence to instructions, content depth,
        and structural coherence. The grader focuses on following specified formats,
        providing comprehensive content, and maintaining logical organization.

        Args:
            query (str): The content generation instruction or query.
            responses (List[str]): The response content(s) to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: Contains a ranked list and explanation.
                - rank (List[int]): Ranking of content by quality
                - reason (str): Explanation of how the ranking was determined
                - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for listwise generation grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = GenerationGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Write a short story about a robot learning to paint",
            ...     responses=[
            ...         "Once upon a time, there was a robot named ART-1 who discovered...",
            ...         "In a future world, robots had many talents, including painting..."
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first story is more creative and follows the prompt better.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
