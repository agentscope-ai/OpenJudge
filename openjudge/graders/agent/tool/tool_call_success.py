# -*- coding: utf-8 -*-
"""
Tool Call Success Grader

Evaluates whether tool calls done by an AI agent includes failures or not.
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TOOL_CALL_SUCCESS_PROMPT_EN = textwrap.dedent(
    """You are an expert evaluator with strong software \
development background. You are required to extract the tool result for every tool call \
then decide for each tool result whether it indicates that the tool call succeeded or failed.

ROLE
====
You are a judge on tool call success who assesses **each tool call made by an AI agent and \
decide if the result of the tool call indicates a success or failure**. You only care about \
technical errors , failures and exceptions , not the business correctness of the tool \
implementation.

You are NOT evaluating:
- The parameters passed to the tool
- The rationale behind choosing this tool
- Whether the successfully returned result from the tool is correct or not business-wise \
given the tool name and definition

You **ARE ONLY** evaluating:
-Whether tool results indicate the presence of a technical error

EVALUATION FRAMEWORK
====================

A tool call is considered successful if:
1. The tool executed without technical errors
2. The tool returned a non-empty result
3. The result is not an error message or exception

A tool call is considered failed if any of the following occurs:
1. The tool result is empty or null
2. The tool result contains explicit error messages
3. The tool result indicates timeout or exception
4. The tool result is an empty object when a populated object is expected

INPUT
=====

TOOL_DEFINITIONS: {tool_definitions}
TOOL_CALLS: {tool_calls}
TOOL_RESPONSES: {tool_responses}
TOOL_CALLS is a list of tool calls that were produced by the AI agent. It includes calls \
together with the result of every tool call.
TOOL_DEFINITIONS is a list of definitions for the tools that were called. This definition \
can contain a description of functionality provided by the tool, the parameters that the \
tool accepts and the expected return of the tool. This definition can contribute to the \
assessment of whether a tool call succeeded or failed.
TOOL_RESPONSES is a list of responses to the tool calls. Each response should be a string.

EXPECTED OUTPUT
===============
Generate a JSON object with the following structure:
```json
{{
  "reason": "Brief explanation of why the tool calls succeeded or failed",
  "score": 1.0 for success or 0.0 for failure
}}
```
"""
).strip()

# Chinese Prompt
TOOL_CALL_SUCCESS_PROMPT_ZH = textwrap.dedent(
    """你是一位具有强大软件开发背景的专家评估员。你需要为每个工具调用提取工具结果，然后判断每个工具结果是否表明工具调用成功或失败。

角色
====
你是工具调用成功性的评判者，负责**评估 AI 智能体进行的每个工具调用，并判断工具调用的结果是否表明成功或失败**。你只关心技术错误、失败和异常，而不关心工具实现的业务正确性。

你不需要评估：
- 传递给工具的参数
- 选择此工具的理由
- 从工具成功返回的结果在业务层面是否正确（考虑工具名称和定义）

你**仅需**评估：
- 工具结果是否表明存在技术错误

评估框架
====================

如果满足以下条件，则认为工具调用成功：
1. 工具执行时没有技术错误
2. 工具返回了非空结果
3. 结果不是错误消息或异常

如果发生以下任何情况，则认为工具调用失败：
1. 工具结果为空或 null
2. 工具结果包含明确的错误消息
3. 工具结果表明超时或异常
4. 当期望返回填充对象时，工具结果是空对象

输入
=====

工具定义：{tool_definitions}
工具调用：{tool_calls}
工具响应：{tool_responses}

工具调用是 AI 智能体生成的工具调用列表。它包括调用以及每个工具调用的结果。
工具定义是被调用工具的定义列表。此定义可以包含工具提供的功能描述、工具接受的参数以及工具的预期返回。此定义可以帮助评估工具调用是否成功或失败。
工具响应是每个工具调用的响应。每个响应应该是一个字符串。

预期输出
===============
生成具有以下结构的 JSON 对象：
```json
{{
  "reason": "工具调用成功或失败的简要说明",
  "score": 成功为 1.0，失败为 0.0
}}
```
"""
).strip()

# Build default template from prompts
DEFAULT_TOOL_CALL_SUCCESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TOOL_CALL_SUCCESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TOOL_CALL_SUCCESS_PROMPT_ZH,
            ),
        ],
    },
)


class ToolCallSuccessGrader(LLMGrader):
    """
    Tool Call Success Grader

    Evaluates whether tool calls done by an AI agent includes failures or not.

    This evaluator focuses solely on tool calls, tool responses, and tool definitions, disregarding user's
    query to the agent, conversation history and agent's final response. Although tool definitions is optional,
    providing them can help the evaluator better understand the context of the tool calls made by the
    agent. Please note that this evaluator validates tool responses for potential technical failures like
    errors, exceptions, timeouts and empty results (only in cases where empty results could indicate a
    failure). It does not assess the correctness of the tool response itself, like mathematical errors and
    unrealistic field values like name="668656".

    Scoring is binary:
    - TRUE (1.0): All tool calls were successful
    - FALSE (0.0): At least one tool call failed

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
        >>> grader = ToolCallSuccessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> tool_definitions = [
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
        >>> tool_responses = [
        ...     "The weather in New York is sunny and 25 degrees Celsius."
        ... ]
        >>> result = asyncio.run(grader.aevaluate(
        ...     tool_definitions=tool_definitions,
        ...     tool_calls=tool_calls,
        ...     tool_responses=tool_responses
        ... ))
        >>> print(result.score)
        1.0
    """

    def __init__(
        self,
        model: Union[BaseChatModel, Dict[str, Any]],
        template: Optional[PromptTemplate] = DEFAULT_TOOL_CALL_SUCCESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """Initialize the ToolCallSuccessGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: Evaluation template. Defaults to DEFAULT_TOOL_CALL_SUCCESS_TEMPLATE.
            language: Language for evaluation prompts (default: LanguageEnum.EN).
        """
        super().__init__(
            name="tool_call_success",
            mode=GraderMode.POINTWISE,
            description="Evaluates whether tool calls done by an AI agent includes failures or not",
            model=model,
            template=template,
            language=language,
        )
        self.template = template or DEFAULT_TOOL_CALL_SUCCESS_TEMPLATE

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
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_responses: Union[str, List[str]],
    ) -> GraderScore:
        """
        Evaluate tool call success

        This method evaluates whether all tool calls were technically successful, focusing only on
        technical aspects like errors, exceptions, timeouts, and empty results. It does not assess
        the correctness of the tool results themselves.

        Args:
            tool_definitions: List of tool definitions whose calls are being evaluated.
                             Each definition typically includes name, description, and parameters.
            tool_calls: List of tool calls to evaluate. Each call should include
                       the tool name and arguments.
            tool_responses: List of tool responses to evaluate. Each response should be a string.

        Returns:
            GraderScore: Score of 1.0 for success or 0.0 for failure

        Example:
            >>> tool_defs = [{"name": "calculator", "description": "Performs calculations"}]
            >>> tool_calls = [
            ...     {"name": "calculator", "arguments": {"expression": "2+2"}}
            ... ]
            >>> tool_responses = [
            ...     "The result of 2+2 is 4."
            ... ]
            >>> result = await grader.aevaluate(
            ...     tool_definitions=tool_defs,
            ...     tool_calls=tool_calls,
            ...     tool_responses=tool_responses
            ... )
        """
        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []
        if not isinstance(tool_responses, list):
            tool_responses = [tool_responses]

        try:
            # Call parent evaluate method with the structured data
            result = await super().aevaluate(
                tool_calls=json.dumps(tool_calls, indent=2),
                tool_definitions=json.dumps(tool_definitions, indent=2),
                tool_responses=json.dumps(tool_responses, indent=2),
            )
            score = result.score
            reason = result.reason
        except Exception as e:
            logger.error(f"Error evaluating tool call success check: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

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
    "ToolCallSuccessGrader",
    "DEFAULT_TOOL_CALL_SUCCESS_TEMPLATE",
]
