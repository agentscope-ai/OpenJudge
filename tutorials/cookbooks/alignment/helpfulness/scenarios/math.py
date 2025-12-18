# -*- coding: utf-8 -*-
"""Math: Solves mathematical problems with accuracy, logical coherence, and proper notation."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Math Listwise System Prompt
MATH_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

MATH_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Math Listwise User Prompt
MATH_USER_PROMPT_EN = """# Task Description
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
Mathematical Accuracy:
    Ensure all calculations, formula applications, and logical steps are error-free, \
as even minor inaccuracies (e.g., arithmetic mistakes, misapplied rules) invalidate \
results despite otherwise correct methodologies.
Logical Coherence:
    Present solutions with clear, sequential reasoning where each step follows logically \
from the previous one, enabling easy verification and understanding of the problem-solving process.
Notational Clarity:
    Use appropriate mathematical notation, symbols, and formatting consistently throughout \
the solution to avoid ambiguity and enhance readability.
Problem-Specific Precision:
    Address all aspects of the given problem completely, providing exact solutions in the \
required format without omitting critical details or providing extraneous information.

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

MATH_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。
我还将为你提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。你必须逐一检查每个候选答案是否违反任何评分标准，并提供你发现的违规原因。这些原因应该作为排名答案的参考。
你可以按自己认为合适的方式组织推理过程，但要尽量保持思维过程简洁。

# 评分标准
数学准确性：
    确保所有计算、公式应用和逻辑步骤都没有错误，因为即使是很小的不准确（例如算术错误、误用规则）也会使结果无效，尽管其他方法是正确的。
逻辑连贯性：
    解决方案要有清晰、连续的推理过程，每一步都要从前一步逻辑推导而来，便于验证和理解解决问题的过程。
符号清晰度：
    在整个解决方案中始终如一地使用适当的数学符号、符号和格式，以避免歧义并提高可读性。
问题特异性精度：
    完全解决给定问题的各个方面，以所需的格式提供确切的解决方案，不省略关键细节或提供多余的信息。

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

MATH_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=MATH_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=MATH_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=MATH_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=MATH_USER_PROMPT_ZH),
        ],
    },
)


class MathGrader(LLMGrader):
    """Math: Solves mathematical problems with accuracy, logical coherence, and proper notation."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = MATH_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the MathGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="math",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Solves mathematical problems with accuracy, " "logical coherence, and proper notation.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the mathematical correctness of the response based on the query.

        Evaluates math responses for accuracy, logical coherence, and notational clarity.
        The grader focuses on error-free calculations, clear reasoning steps, and
        appropriate mathematical notation.

        Args:
            query (str): The mathematical problem or query to evaluate.
            responses (List[str]): The mathematical solutions to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> # Example for listwise math grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = MathGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Solve for x: 2x + 5 = 15",
            ...     responses=[
            ...         "2x + 5 = 15\\n2x = 15 - 5\\n2x = 10\\nx = 5",
            ...         "2x + 5 = 15\\n2x = 10\\nx = 5"
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first answer shows all steps clearly while the second omits one step.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
