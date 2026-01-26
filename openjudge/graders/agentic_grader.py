# -*- coding: utf-8 -*-
"""
Agentic grader implementation for tool-augmented evaluation.

This module provides the AgenticGrader class, which uses ReAct-style (Reasoning + Acting)
autonomous tool calling to evaluate model responses. The grader supports multi-turn
interactions where the LLM decides which tools to call and when to produce final judgment.

The module implements OpenAI function calling for seamless tool integration and provides
flexible result parsing for both pointwise scoring and listwise ranking.

Classes:
    BaseTool: Abstract base class for implementing custom tools.
    ToolResult: Pydantic model for tool execution results.
    AgenticGrader: Main class for ReAct-style tool-augmented evaluation.
"""

import asyncio
import json
import os
import re
import textwrap
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderRank, GraderScore
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# ============================================================================
# Tool System
# ============================================================================


class ToolResult(BaseModel):
    """Result returned by a tool execution.

    Attributes:
        success: Whether the tool execution was successful.
        output: The output data from the tool execution.
        error: Error message if the execution failed.
        metadata: Additional metadata from the execution.
    """

    success: bool = Field(default=True, description="Whether the tool execution was successful")
    output: Any = Field(default=None, description="The output data from the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(ABC):
    """Abstract base class for tools. Uses OpenAI function calling format.

    Example:
        >>> class WebSearchTool(BaseTool):
        ...     schema = {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "web_search",
        ...             "description": "Search the web for information",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "query": {"type": "string", "description": "Search query"}
        ...                 },
        ...                 "required": ["query"]
        ...             }
        ...         }
        ...     }
        ...
        ...     async def aexecute(self, query: str, **kwargs) -> ToolResult:
        ...         # Perform search
        ...         return ToolResult(success=True, output="search results...")
    """

    # OpenAI function calling format - subclasses MUST override
    schema: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "base_tool",
            "description": "Base tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    @property
    def name(self) -> str:
        """Get tool name from schema."""
        return self.schema.get("function", {}).get("name", "unknown")

    @property
    def description(self) -> str:
        """Get tool description from schema."""
        return self.schema.get("function", {}).get("description", "")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters from schema."""
        return self.schema.get("function", {}).get("parameters", {})

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""

    def execute(self, **kwargs: Any) -> ToolResult:
        """Synchronous wrapper for tool execution."""
        return asyncio.run(self.aexecute(**kwargs))


# ============================================================================
# Agentic Grader (ReAct Style)
# ============================================================================


class AgenticGrader(BaseGrader):
    """ReAct-style agentic grader using OpenAI function calling.

    The LLM autonomously decides which tools to call in a loop until it
    produces a final judgment. This follows the standard ReAct pattern:

        Thought -> Action -> Observation -> Thought -> ... -> Final Answer

    Implemented via OpenAI function calling:
        1. LLM receives task + available tools
        2. LLM returns tool_calls (or final answer)
        3. Execute tools, append results to messages
        4. Repeat until LLM returns content without tool_calls

    Example:
        >>> grader = AgenticGrader(
        ...     model=model,
        ...     tools=[WebSearchTool(), CodeExecutionTool()],
        ...     template=PromptTemplate(...),
        ... )
        >>> result = await grader.aevaluate(query="...", response="...")
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        tools: Optional[List[BaseTool]] = None,
        template: str | dict | list | PromptTemplate | None = None,
        name: str = "agentic_grader",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "ReAct-style agentic grader",
        language: LanguageEnum | str | None = None,
        max_iterations: int = 10,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Initialize AgenticGrader.

        Args:
            model: LLM for reasoning and tool calling. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            tools: List of available tools.
            template: Template for generating prompts. Can be a str, list, dict or
                     PromptTemplate object. If None, uses a default evaluation prompt.
            name: Grader name.
            mode: POINTWISE or LISTWISE.
            description: Grader description.
            language: Language for prompts. Can be LanguageEnum, string, or None.
                     If None, defaults to environment variable LANGUAGE or "en".
            max_iterations: Max tool-calling iterations to prevent infinite loops.
            callback: Callback function for processing model response metadata.
            **kwargs: Additional keyword arguments passed to template rendering.
        """
        super().__init__(name=name, mode=mode, description=description, **kwargs)

        # Handle language parameter
        if not language:
            language = os.environ.get("LANGUAGE", "en")

        if isinstance(language, str):
            # Convert string to LanguageEnum
            self.language = (
                LanguageEnum(language) if language in [item.value for item in LanguageEnum] else LanguageEnum.EN
            )
        else:
            self.language = language

        # Handle template parameter
        if template is None:
            self.template = None  # Will use fallback in _build_messages
        elif isinstance(template, str):
            self.template = PromptTemplate(
                messages={
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation assistant. "
                            "Please evaluate according to the user's requirements.",
                        ),
                        ChatMessage(
                            role="user",
                            content=textwrap.dedent(template),
                        ),
                    ],
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="system",
                            content="你是个专业的评估助手，请你根据用户要求进行评估。",
                        ),
                        ChatMessage(
                            role="user",
                            content=textwrap.dedent(template),
                        ),
                    ],
                },
            )
        elif isinstance(template, PromptTemplate):
            self.template = template
        elif isinstance(template, list):
            # Support list of message dicts or ChatMessage objects
            self.template = PromptTemplate.from_prompt(template)
        elif isinstance(template, dict):
            self.template = PromptTemplate(**template)
        else:
            raise ValueError("Template must be a str, list, dict or PromptTemplate object")

        # Initialize model
        if isinstance(model, dict):
            self.model = OpenAIChatModel(**model)
        else:
            self.model = model

        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations
        self.callback = callback

    # -------------------------------------------------------------------------
    # Tool Management
    # -------------------------------------------------------------------------

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self.tools.get(name)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schema for all tools."""
        return [tool.schema for tool in self.tools.values()]

    # -------------------------------------------------------------------------
    # Core ReAct Loop
    # -------------------------------------------------------------------------

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call and return the result message.

        Args:
            tool_call: OpenAI tool_call object with id, function.name, function.arguments

        Returns:
            Tool result message in OpenAI format.
        """
        tool_call_id = tool_call.get("id", str(uuid.uuid4()))
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        # Parse arguments
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {}

        # Get and execute tool
        tool = self.get_tool(tool_name)
        if not tool:
            content = f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
        else:
            try:
                result = await tool.aexecute(**arguments)
                if result.success:
                    content = str(result.output) if result.output else "Success (no output)"
                else:
                    # Prefer output over error for detailed failure info
                    content = str(result.output) if result.output else f"Error: {result.error}"
            except Exception as e:
                content = f"Execution error: {str(e)}"

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    async def _react_loop(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Run the ReAct loop until LLM produces final answer.

        Args:
            messages: Initial messages (system + user).
            tools_schema: OpenAI function calling schema.

        Returns:
            Tuple of (final_content, full_messages_history).
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"ReAct iteration {iteration}/{self.max_iterations}")

            # Call LLM with tools
            try:
                response = await self.model.achat(
                    messages=messages,
                    tools=tools_schema if tools_schema else None,
                    callback=self.callback,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

            # Handle streaming response
            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    response = chunk

            # Extract response content and tool_calls
            content = getattr(response, "content", None) or ""
            tool_calls = getattr(response, "tool_calls", None) or []

            # Build assistant message
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            if tool_calls:
                # Convert tool_calls to serializable format
                assistant_msg["tool_calls"] = [
                    {
                        "id": (
                            tc.get("id", str(uuid.uuid4()))
                            if isinstance(tc, dict)
                            else getattr(tc, "id", str(uuid.uuid4()))
                        ),
                        "type": "function",
                        "function": {
                            "name": (
                                tc.get("function", {}).get("name", "")
                                if isinstance(tc, dict)
                                else getattr(tc.function, "name", "")
                            ),
                            "arguments": (
                                tc.get("function", {}).get("arguments", "{}")
                                if isinstance(tc, dict)
                                else getattr(tc.function, "arguments", "{}")
                            ),
                        },
                    }
                    for tc in tool_calls
                ]

            messages.append(assistant_msg)

            # If no tool calls, we have the final answer
            if not tool_calls:
                logger.debug(f"ReAct completed after {iteration} iterations")
                return content, messages

            # Execute all tool calls
            for tc in assistant_msg.get("tool_calls", []):
                tool_result_msg = await self._execute_tool_call(tc)
                messages.append(tool_result_msg)
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", "{}")
                tool_output = tool_result_msg["content"]
                # Log full tool call details for debugging
                logger.debug(f"Tool call: {tool_name}\n" f"  Args: {tool_args}\n" f"  Result: {tool_output}")

        # Max iterations reached
        logger.warning(f"ReAct reached max iterations ({self.max_iterations})")
        return content, messages

    # -------------------------------------------------------------------------
    # Evaluation Interface
    # -------------------------------------------------------------------------

    def _build_messages(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Build initial messages from template.

        Args:
            **kwargs: Parameters to be filled into the template.

        Returns:
            List of message dictionaries in OpenAI format.
        """
        # Merge kwargs: self.kwargs first, then call-time kwargs override
        params = {**self.kwargs}
        params.update(kwargs)

        if self.template:
            # Use template to generate messages
            messages = self.template.format(language=self.language, **params)
            return [msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in messages]
        else:
            # Fallback: simple default prompt
            query = params.get("query", "")
            response = params.get("response", "")
            return [
                {
                    "role": "user",
                    "content": f"Please evaluate the following response:\n\n"
                    f"**Query/Task:**\n{query}\n\n"
                    f"**Response to evaluate:**\n{response}\n\n"
                    f"Use the available tools to gather evidence if needed, "
                    f"then provide your final evaluation.",
                }
            ]

    def _parse_result(self, llm_output: str) -> Union[GraderScore, GraderRank]:
        """Parse LLM output to GraderScore/GraderRank.

        Attempts to extract structured data from LLM output using multiple strategies:
        1. Try to parse JSON (supports nested JSON with recursive matching)
        2. Fall back to regex extraction for score/rank patterns
        3. Use the raw output as reason if parsing fails

        Args:
            llm_output: Raw LLM output string.

        Returns:
            GraderScore or GraderRank depending on mode.
        """
        data: Dict[str, Any] = {}

        # Strategy 1: Try to extract JSON from text (supports nested JSON)
        try:
            # Find all potential JSON objects using balanced brace matching
            json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
            json_matches = re.findall(json_pattern, llm_output, re.DOTALL)

            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    # Check if this JSON has the fields we need
                    if "score" in parsed or "rank" in parsed:
                        data = parsed
                        break
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to parse JSON from LLM output: {e}")

        # Strategy 2: Fall back to regex extraction if no valid JSON found
        if not data:
            if self.mode == GraderMode.POINTWISE:
                # Try to extract score from text patterns
                score_patterns = [
                    r"(?:score|rating|分数)[:\s]*(\d+(?:\.\d+)?)",  # score: 4 or 分数: 4
                    r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+",  # 4 out of 5 or 4/5
                    r"(?:give|assign|rate)[^0-9]*(\d+(?:\.\d+)?)",  # give it a 4
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, llm_output, re.IGNORECASE)
                    if match:
                        data["score"] = float(match.group(1))
                        break
            else:
                # Try to extract rank from text patterns (e.g., [1, 2, 3] or rank: 2, 1, 3)
                rank_patterns = [
                    r"rank[:\s]*\[([^\]]+)\]",  # rank: [1, 2, 3]
                    r"\[(\d+(?:\s*,\s*\d+)+)\]",  # [1, 2, 3]
                    r"rank[:\s]*([\d,\s]+)",  # rank: 1, 2, 3
                ]
                for pattern in rank_patterns:
                    match = re.search(pattern, llm_output, re.IGNORECASE)
                    if match:
                        try:
                            rank_str = match.group(1)
                            data["rank"] = [int(x.strip()) for x in rank_str.split(",")]
                            break
                        except ValueError:
                            continue

        # Build result based on mode
        if self.mode == GraderMode.POINTWISE:
            score = data.get("score")
            if score is None:
                raise ValueError(f"Failed to extract 'score' from LLM output: {llm_output}")
            reason = data.get("reason", llm_output)
            metadata = {k: v for k, v in data.items() if k not in ("score", "reason")}

            return GraderScore(name=self.name, score=float(score), reason=reason, metadata=metadata)
        else:
            rank = data.get("rank")
            if rank is None:
                raise ValueError(f"Failed to extract 'rank' from LLM output: {llm_output}")
            reason = data.get("reason", llm_output)
            metadata = {k: v for k, v in data.items() if k not in ("rank", "reason")}

            return GraderRank(name=self.name, rank=rank, reason=reason, metadata=metadata)

    async def aevaluate(
        self,
        query: str = "",
        response: str = "",
        **kwargs: Any,
    ) -> Union[GraderScore, GraderRank]:
        """Evaluate using ReAct-style tool-augmented LLM.

        The LLM autonomously decides which tools to call and when to
        produce the final judgment.

        Args:
            query: The original query/task.
            response: The response to evaluate.
            **kwargs: Additional context passed to the template.

        Returns:
            GraderScore: In POINTWISE mode, contains score, reason, and metadata.
            GraderRank: In LISTWISE mode, contains rank list, reason, and metadata.

        Raises:
            Exception: If evaluation fails for any reason.

        Example:
            >>> grader = AgenticGrader(
            ...     model=OpenAIChatModel(model="gpt-4"),
            ...     tools=[WebSearchTool()],
            ... )
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="The capital of France is Paris."
            ... )
            >>> print(result.score, result.reason)
        """
        start_time = time.time()

        # Build initial messages from template (merges self.kwargs with call-time kwargs)
        messages = self._build_messages(query=query, response=response, **kwargs)

        # Get tools schema
        tools_schema = self.get_tools_schema()

        # Run ReAct loop
        final_content, full_messages = await self._react_loop(messages, tools_schema)

        # Parse result from LLM text output
        result = self._parse_result(final_content)

        # Count tool calls from message history
        tool_call_count = sum(len(msg.get("tool_calls", [])) for msg in full_messages if msg.get("role") == "assistant")

        # Add metadata (merge with any existing metadata from parsed)
        result.metadata = {
            **result.metadata,
            "total_time": time.time() - start_time,
            "tool_calls": tool_call_count,
            "iterations": len([m for m in full_messages if m.get("role") == "assistant"]),
            "messages": full_messages,  # Full conversation for debugging
        }

        return result

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return the docstring of the aevaluate method to explain how AgenticGrader works."""
        return {"aevaluate": AgenticGrader.aevaluate.__doc__, "prompt": {}}

    def to_dict(self) -> dict:
        """Convert the AgenticGrader to a dictionary representation.

        This method serializes the AgenticGrader's properties (name, mode, description, template,
        tools, max_iterations) into a dictionary. The mode is converted to its string value,
        and the template is converted to a dictionary if it is a PromptTemplate.

        Returns:
            A dictionary containing the serialized AgenticGrader information.
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "template": (self.template.model_dump() if isinstance(self.template, PromptTemplate) else self.template),
            "tools": list(self.tools.keys()),
            "max_iterations": self.max_iterations,
            **self.kwargs,
        }

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "AgenticGrader":
        """Create an AgenticGrader from a configuration dictionary.

        This class method creates a new AgenticGrader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) and AgenticGrader-specific
        properties (template, model, tools, max_iterations) from the config.

        Args:
            config: A dictionary containing the AgenticGrader configuration.
                   Expected keys include 'name', 'mode', 'description', 'template',
                   'model', 'tools', 'max_iterations', and any additional parameters.

        Returns:
            A new AgenticGrader instance.
        """
        # Make a copy to avoid modifying the original config
        config = config.copy()

        # Extract standard grader properties
        name = config.pop("name", "agentic_grader")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "ReAct-style agentic grader")

        # Extract AgenticGrader-specific properties
        template = config.pop("template", None)
        model = config.pop("model", {})
        tools = config.pop("tools", None)
        max_iterations = config.pop("max_iterations", 10)

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            template=template,
            model=model,
            tools=tools,
            max_iterations=max_iterations,
            **config,
        )
