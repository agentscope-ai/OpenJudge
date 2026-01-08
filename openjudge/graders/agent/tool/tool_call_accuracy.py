# -*- coding: utf-8 -*-
"""
Tool Call Accuracy Grader

Evaluates the accuracy of tool calls made by an agent.
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TOOL_CALL_ACCURACY_PROMPT_EN = textwrap.dedent(
    """# Instruction
## Goal
Your are an expert in evaluating the accuracy of a tool call considering relevance and \
potential usefulness including syntactic and semantic correctness of a proposed tool call \
from an intelligent system based on provided definition and data. Your goal will involve \
answering the questions below using the information provided.

# Definition
**Tool Call Accuracy** refers to the overall effectiveness of TOOL CALLS made by an \
agent in response to a user's query within an ongoing CONVERSATION.

# EVALUATION CRITERIA
Evaluate based on these factors:
1. **Tool Relevance**: Do the tool call appropriately address the user's query?
2. **Parameter Correctness**: Are all parameter values extracted from or reasonably inferred \
    from the CONVERSATION?

# Ratings
- Score 1: The tool calls were irrelevant to answer the user query, all names of tool calls could not be found as one of the function names in the tool definitions, or parameters of tool calls were incorrect.
- Score 2: The tool calls were irrelevant to answer the user query, some names of tool calls could not be found as one of the function names in tool definitions, or parameters of tool calls were incorrect.
- Score 3: The tool calls were partially relevant to answer the user query, and the description in tool definitions, all names of tool calls could be found as one of the function names in tool definitions, and parameters of tool calls were correct but irrelevant with the description in tool definitions.
- Score 4: The tool calls were fairly relevant to answer the user query, all names of tool calls could be found as one of the function names in tool definitions, and parameters of tool calls were correct and relevant with the description in tool definition.
- Score 5: The tool calls were fully relevant to answer the user query, all names of tool calls could be found as one of the function names in tool definitions, and parameters of tool calls were correct and relevant with the description in tool definition.

# Data
CONVERSATION : {query}
TOOL CALLS TO BE EVALUATED: {tool_calls}
TOOL DEFINITIONS: {tool_definitions}

# Tasks
Please provide your evaluation for the tool calls in relation to the user query and tool definitions.
Your output should be a JSON object with the following format:
```json
{{
    "score": [Tool Call Accuracy Score],
    "reason": [Reason for the score],
}}
```
"""
).strip()

# Chinese Prompt
TOOL_CALL_ACCURACY_PROMPT_ZH = textwrap.dedent(
    """# 指令
## 目标
你是评估工具调用准确性的专家，需要考虑相关性和潜在有用性，包括基于提供的定义和数据，对智能系统提出的工具调用的语法和语义正确性进行评估。你的目标是使用提供的信息回答以下问题。

# 定义
**工具调用准确性**是指智能体在正在进行的对话中响应用户查询所做的工具调用的整体有效性。

# 评估标准
基于以下因素进行评估：
1. **工具相关性**：工具调用是否适当地解决了用户的查询？
2. **参数正确性**：所有参数值是否从对话中提取或合理推断？

# 评分
- 分数 1：工具调用与回答用户问题无关，所有工具调用名称均未在工具定义中找到对应的函数名称，或者工具调用的参数不正确。
- 分数 2：工具调用与回答用户问题无关，部分工具调用名称未在工具定义中找到对应的函数名称，或者工具调用的参数不正确。
- 分数 3：工具调用与回答用户问题部分相关，工具定义中的描述与用户问题相关，所有工具调用名称均在工具定义中找到对应的函数名称，并且工具调用的参数正确，但与工具定义中的描述无关。
- 分数 4：工具调用与回答用户问题基本相关，所有工具调用名称均在工具定义中找到对应的函数名称，并且工具调用的参数正确，与工具定义中的描述相关。
- 分数 5：工具调用与回答用户查询完全相关，所有工具调用的名称都可以在工具定义中找到，并且工具调用的参数与工具定义中的描述一致且正确。

# 数据
对话：{query}
待评估的工具调用：{tool_calls}
工具定义：{tool_definitions}

# 任务
请提供对工具调用相对于用户查询和工具定义的评估。
你的输出应该是具有以下格式的 JSON 对象：
```json
{{
    "score": [工具调用准确性分数],
    "reason": [分数的原因],
}}
```
"""
).strip()

# Build default template from prompts
DEFAULT_TOOL_CALL_ACCURACY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TOOL_CALL_ACCURACY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TOOL_CALL_ACCURACY_PROMPT_ZH,
            ),
        ],
    },
)


class ToolCallAccuracyGrader(LLMGrader):
    """
    Tool Call Accuracy Grader

    Evaluates the accuracy of tool calls made by an agent.

    The ToolCallAccuracyGrader assesses how accurately an AI uses tools by examining:
    - Relevance to the conversation
    - Parameter correctness according to tool definitions
    - Parameter value extraction from the conversation

    The evaluator uses a scoring rubric of 1 to 5:
    - Score 1: The tool calls are irrelevant
    - Score 2: The tool calls are partially relevant, but not enough tools were called or the
               parameters were not correctly passed
    - Score 3: The tool calls are relevant, but there were unnecessary, excessive tool calls made
    - Score 4: The tool calls are relevant, but some tools returned errors and agent retried calling
               them again and succeeded
    - Score 5: The tool calls are relevant, and all parameters were correctly passed

    This evaluation focuses on measuring whether tool calls meaningfully contribute to addressing
    user needs while properly following tool definitions and using information present in the
    conversation history.

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = ToolCallAccuracyGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": "What's the weather like in New York?"
        ...     }
        ... ]
        ... tool_definitions = [
        ...     {
        ...         "name": "get_weather",
        ...         "description": "Get weather information for a location",
        ...         "parameters": {
        ...             "location": "City name"
        ...         }
        ...     }
        ... ]
        ... tool_calls = [
        ...     {
        ...         "name": "get_weather",
        ...         "arguments": {"location": "New York"}
        ...     }
        ... ]
        >>> result = asyncio.run(grader.aevaluate(
        ...     query=conversation,
        ...     tool_definitions=tool_definitions,
        ...     tool_calls=tool_calls
        ... ))
        >>> print(result.score)
        5.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_TOOL_CALL_ACCURACY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """Initialize the ToolCallAccuracyGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: Evaluation template. Defaults to DEFAULT_TOOL_CALL_ACCURACY_TEMPLATE.
            language: Language for evaluation prompts (default: LanguageEnum.EN).
        """
        super().__init__(
            name="tool_call_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluates the accuracy of tool calls made by an agent",
            model=model,
            template=template or DEFAULT_TOOL_CALL_ACCURACY_TEMPLATE,
            language=language,
        )

    def _parse_tools_from_response(
        self,
        response: str,
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from the response.

        Args:
            response: The response string to extract tool calls from.

        Returns:
            List of parsed tool calls.
        """
        tool_calls = []

        # Pattern to match tool calls in JSON format
        tool_call_pattern = r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

        return tool_calls

    async def aevaluate(
        self,
        query: str | List[Dict[str, Any]],
        tool_definitions: Dict[str, Any] | List[Dict[str, Any]],
        tool_calls: Dict[str, Any] | List[Dict[str, Any]] | None = None,
        response: str | List[Dict[str, Any]] | None = None,
    ) -> GraderScore | GraderError:
        """
        Evaluate tool call accuracy

        This method evaluates the accuracy of tool calls based on multiple criteria including
        relevance, parameter correctness, completeness, efficiency, and execution success.
        It assigns a score from 1 to 5 based on how well the tool calls address the user's query.

        Args:
            query: Query or Chat history up to the message that has the tool call being evaluated.
                  Can be a string for simple queries or a list of message dictionaries for
                  conversation history.
            tool_definitions: List of tool definitions whose calls are being evaluated.
                             Each definition includes name, description, and parameters information.
            tool_calls: Optional List of tool calls to evaluate. If not provided, response should be
                       provided and should contain tool call(s).
            response: Optional response to be evaluated alongside the tool calls.
                     If provided and tool_calls parameter is not provided, all tool calls in
                     response will be evaluated.
                     If both response and tool_calls parameters are provided, only the tool calls in
                     tool_calls parameter will be evaluated.

        Returns:
            GraderScore: Score from 1.0 to 5.0 indicating tool call accuracy

        Example:
            >>> conversation = [
            ...     {"role": "user", "content": "What's the weather like in New York?"}
            ... ]
            >>> tool_defs = [
            ...     {
            ...         "name": "get_weather",
            ...         "description": "Get weather information for a location",
            ...         "parameters": {"location": "City name"}
            ...     }
            ... ]
            >>> tool_calls = [
            ...     {
            ...         "name": "get_weather",
            ...         "arguments": {"location": "New York"}
            ...     }
            ... ]
            >>> result = await grader.aevaluate(
            ...     query=conversation,
            ...     tool_definitions=tool_defs,
            ...     tool_calls=tool_calls
            ... )
        """
        # Handle tool calls extraction from response if needed
        if response and not tool_calls:
            parsed_tool_calls = self._parse_tools_from_response(str(response))
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        # Check if we have tool calls to evaluate
        if not tool_calls:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No tool calls found in the response to evaluate",
                metadata={
                    "error": "No tool calls provided or found in response",
                },
            )

        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        try:
            # Call parent evaluate method with the structured data
            result = await super().aevaluate(
                query=json.dumps(query, indent=2, ensure_ascii=False),
                tool_calls=json.dumps(tool_calls, indent=2, ensure_ascii=False),
                tool_definitions=json.dumps(tool_definitions, indent=2, ensure_ascii=False),
            )
            score = max(1.0, min(5.0, result.score))
            reason = result.reason
        except Exception as e:
            logger.error(f"Error evaluating tool call accuracy check: {e}")
            reason = f"Evaluation error: {str(e)}"
            return GraderError(
                name=self.name,
                error=reason,
            )

        # Prepare metadata
        metadata = {
            "raw_score": score,
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ToolCallAccuracyGrader",
    "DEFAULT_TOOL_CALL_ACCURACY_TEMPLATE",
]
