# -*- coding: utf-8 -*-
"""Classification: Accurately categorizes input content into predefined classes or labels."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Classification Listwise System Prompt
CLASSIFICATION_SYSTEM_PROMPT_EN = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

CLASSIFICATION_SYSTEM_PROMPT_ZH = "你是一个擅长奖励评估的助手。请根据给定的提示词进行奖励判断。"

# Classification Listwise User Prompt
CLASSIFICATION_USER_PROMPT_EN = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate responses. Your task is to select the single best answer \
from the candidates.

# Rubrics
Accuracy and Correctness:
    Prioritize selecting the most appropriate category based on the input \
content, ensuring the classification decision aligns with the dominant \
themes, sentiment, or subject matter.
Clarity of Reasoning:
    Provide clear justification for the classification decision, referencing \
specific elements from the input that support the chosen category.
Handling Ambiguity:
    Appropriately address ambiguous cases by either selecting the most probable \
category with explanation or indicating uncertainty when no clear \
classification is possible.
Consistency:
    Apply consistent classification criteria across similar types of input \
content, maintaining stable categorization standards.

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

CLASSIFICATION_USER_PROMPT_ZH = """# 任务描述
你的角色是一名专业的评估专家。我将向你提供一个问题和几个候选答案。你的任务是从候选答案中选择最佳答案。

# 评分标准
准确性和正确性：
    优先根据输入内容选择最合适的类别，确保分类决策与主要内容、情感或主题保持一致。
推理清晰度：
    为分类决策提供明确的理由，引用支持所选类别的输入中的具体元素。
处理歧义：
    适当地处理模糊情况，要么选择最可能的类别并加以解释，要么在无法明确分类时表明不确定性。
一致性：
    在相似类型的输入内容中应用一致的分类标准，保持稳定的分类标准。

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

CLASSIFICATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="system", content=CLASSIFICATION_SYSTEM_PROMPT_EN),
            ChatMessage(role="user", content=CLASSIFICATION_USER_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="system", content=CLASSIFICATION_SYSTEM_PROMPT_ZH),
            ChatMessage(role="user", content=CLASSIFICATION_USER_PROMPT_ZH),
        ],
    },
)


class ClassificationGrader(LLMGrader):
    """Classification: Accurately categorizes input content into predefined classes or labels."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = CLASSIFICATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the ClassificationGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="classification",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Accurately categorizes input content into predefined classes or labels.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the classification response based on the query.

        Evaluates classification responses for their ability to accurately
        categorize input content into predefined classes or labels. The grader
        focuses on accurate category assignment, clear decision justification,
        and appropriate handling of ambiguous cases.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The list of responses to evaluate and rank.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result containing the rank of responses and reasoning.

        Example:
            >>> grader = ClassificationGrader()
            >>> result = await grader.aevaluate(
            ...     query="This movie was fantastic! I loved every minute of it.",
            ...     responses=["Positive", "Negative", "Neutral"]
            ... )
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
