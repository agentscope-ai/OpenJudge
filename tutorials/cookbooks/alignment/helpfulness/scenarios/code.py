# -*- coding: utf-8 -*-
"""Code: Generates correct, efficient, and readable code solutions to programming problems."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Code Listwise System Prompt
CODE_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

CODE_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Code Listwise User Prompt
CODE_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Functional Correctness:
    The code must correctly implement the required functionality without syntax \
errors or logical flaws. It should pass basic test cases related to the \
problem.
Code Quality and Readability:
    The code should be well-structured, readable, and follow common programming \
conventions. Variable names should be meaningful, and complex logic should \
be appropriately commented.
Efficiency and Optimization:
    The solution should be reasonably efficient in terms of time and space \
complexity. Avoid unnecessary computations or memory usage.
Language-Specific Best Practices:
    The code should follow language-specific conventions and idioms, making it \
appear as if written by an experienced developer in that language.
Error Handling:
    Appropriate error handling should be implemented for edge cases and invalid \
inputs, making the code robust and production-ready.

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

CODE_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
功能性正确性：
    代码必须正确实现所需的功能，没有语法错误或逻辑缺陷。它应该通过与问题相关的基本测试用例。
代码质量和可读性：
    代码应该结构良好、可读性强，并遵循常见的编程约定。变量名应该有意义，复杂的逻辑应该有适当的注释。
效率和优化：
    解决方案在时间和空间复杂度方面应该是合理的。避免不必要的计算或内存使用。
语言特定的最佳实践：
    代码应遵循特定语言的约定和习惯用法，使其看起来像是由该语言的经验丰富的开发者编写的。
错误处理：
    应为边缘情况和无效输入实现适当的错误处理，使代码健壮且可用于生产环境。

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

CODE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=CODE_SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=CODE_USER_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=CODE_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=CODE_USER_PROMPT_ZH,
            ),
        ],
    },
)


class CodeGrader(LLMGrader):
    """Code: Generates correct, efficient, and readable code solutions to programming problems."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = CODE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the CodeGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="code",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Generates correct, efficient, and readable code solutions to " "programming problems.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the code response based on the query.

        Evaluates code responses for their correctness, efficiency, readability,
        and adherence to best practices. The grader focuses on functional correctness,
        code quality, efficiency, and language-specific conventions.

        Args:
            query (str): The programming problem or query to evaluate.
            responses (List[str]): The code solutions to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: Contains a ranked list and explanation.
                - rank (List[int]): Ranking of code solutions by quality
                - reason (str): Explanation of how the ranking was determined
                - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for listwise code grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> grader = CodeGrader(model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Write a function to calculate the factorial of a number",
            ...     responses=[
            ...         "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
            ...         "def fact(n):\\n    result = 1\\n    for i in range(1, n+1):"
            ...         "\\n        result *= i\\n    return result"
            ...     ]
            ... ))
            >>> print(result.rank, result.reason)
            [1, 2] The first answer uses recursion which is more elegant for factorial calculation.
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
