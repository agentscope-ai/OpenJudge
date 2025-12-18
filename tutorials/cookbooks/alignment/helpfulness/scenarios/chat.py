# -*- coding: utf-8 -*-
"""Chat: Maintains natural, engaging conversations while providing accurate information."""
from typing import Any, List

from rm_gallery.core.graders.base_grader import GraderMode, GraderRank
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# Chat Rank System Prompt
CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant skilled in reward evaluation. "
    "Please make reward judgments based on the given prompt words."
)

# Chat Rank User Prompt (English)
CHAT_USER_PROMPT = """# Task Description
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
Address Core Argument/Intent Directly:
    Prioritize engaging with the user's central claim, perspective, or question \
explicitly, ensuring responses align with their stated goals or concerns \
rather than diverging into tangential topics.
Provide Actionable, Context-Specific Guidance:
    Offer concrete, practical steps or solutions tailored to the user's unique \
situation, balancing clarity with adaptability to empower informed decisions \
or actions.
Ensure Factual Accuracy and Contextual Nuance:
    Correct misconceptions, clarify complexities, and ground responses in precise \
details or evidence while avoiding oversimplification or speculative \
interpretations.

# Query
{query}

# Responses
{responses}

# Output Requirement
```json
{
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}
```
"""

# Chat Rank User Prompt (Chinese)
CHAT_SYSTEM_PROMPT_ZH = (
    "你是一位专业评估专家，请根据给定的问题，从候选答案中选择最佳答案。"
    "请根据以下评分标准对每个候选答案进行评价，并给出理由。"
)

CHAT_USER_PROMPT_ZH = """# 任务描述
你的角色是专业评估专家。我将为您提供一个问题和几个候选答案。您的任务是从候选人中选择最佳答案。
我还将为您提供一组评分标准，列在#评分标准标题下。这些评分标准按重要性从高到低排列。\
您必须依次检查每个候选答案，看它是否违反任何评分标准，并提供违规原因。这些原因应该作为排名答案的参考。
您可以按自己认为合适的方式组织推理，但要尽量保持思维过程简洁。

# 评分标准
直接解决核心论点/意图:
    优先明确处理用户的中心主张、观点或问题，确保回应与其 stated的目标或关注点一致，\
而不是偏离到无关的话题上。
提供可行的、情境特定的指导:
    提供针对用户独特情况的 concrete、实用步骤或解决方案，平衡清晰度与适应性，以赋予知情决策或行动的能力。
确保事实准确性和情境细微差别:
    纠正误解，澄清复杂性，并在精确的细节或证据中落实回应，同时避免过度简化或推测性解释。

# 查询
{query}

# 答案
{responses}

# 输出要求
```json
{
    "rank": ["答案在列表中的排名分数"]
    "reason": "得分原因"
}
```
"""

CHAT_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=CHAT_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role="user",
                content=CHAT_USER_PROMPT,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=CHAT_SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=CHAT_USER_PROMPT_ZH,
            ),
        ],
    },
)


class ChatGrader(LLMGrader):
    """Chat: Maintains natural, engaging conversations while providing accurate information."""

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: PromptTemplate = CHAT_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs: Any,
    ):
        """Initialize the ChatGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to LISTWISE.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="chat",
            mode=GraderMode.LISTWISE,
            model=model,
            template=template,
            language=language,
            description="Maintains natural, engaging conversations while providing accurate information.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        responses: List[str],
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate the quality of the chat response based on the query.

        Evaluates chat responses for their ability to simulate human conversation
        and communicate effectively across various topics. The grader emphasizes
        coherence and natural flow of interaction while ensuring responses are
        contextually appropriate and helpful.

        Args:
            query (str): The query to evaluate.
            responses (List[str]): The answer(s) to evaluate. For LISTWISE mode,
                this should be a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderRank: The evaluation result.

                Each GraderRank contains:
                    - rank: A ranked list of the responses
                    - reason: Explanation of how the ranking was determined
                    - metadata: Optional additional information from the evaluation

        Example:
            >>> grader = ChatGrader()
            >>> result = await grader.aevaluate(
            ...     query="Hi, how are you today?",
            ...     answer=["I'm doing well, thank you for asking! How can I assist you?",
            ...             "Fine. What do you want?"]
            ... )
        """
        responses_str = "\n".join([f"{i}. {answer}" for i, answer in enumerate(responses, start=1)])
        return await super().aevaluate(query=query, responses=responses_str, **kwargs)
