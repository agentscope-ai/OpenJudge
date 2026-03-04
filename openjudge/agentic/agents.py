# -*- coding: utf-8 -*-
"""
Agent system for agentic evaluation.

This module provides the agent infrastructure for OpenJudge's agentic evaluation,
including the base agent class and built-in ReAct agent implementation. An agent
is responsible for the "thinking" part of agentic evaluation - it takes messages
and tools, runs the reasoning loop, and returns the final answer.

The agent abstraction allows different reasoning strategies (ReAct, CoT, Tree-of-Thought,
etc.) to be plugged into the same AgenticGrader without changing the grader's code.

Classes:
    AgentResult: Standardized result format for agent execution.
    BaseAgent: Abstract base class for all agents.
    ReActAgent: Built-in ReAct agent implementation using OpenAI function calling.

Example:
    >>> from openjudge.agentic import ReActAgent
    >>>
    >>> # Create agent with model instance
    >>> agent = ReActAgent(model=my_model, tools=[tool1, tool2])
    >>>
    >>> # Or with dict config (convenience)
    >>> agent = ReActAgent(
    ...     model={"model": "gpt-4", "api_key": "..."},
    ...     tools=[tool1, tool2],
    ... )
    >>>
    >>> result = await agent.arun(messages)
    >>> print(result.content)
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel

from .tools import BaseTool

__all__ = [
    "AgentResult",
    "BaseAgent",
    "ReActAgent",
]


class AgentResult(BaseModel):
    """Standardized result format for agent execution.

    This class provides a consistent interface for all agent outputs in OpenJudge.
    It contains the final answer along with metadata about the execution process,
    which is useful for debugging and analysis.

    Attributes:
        content: The final text content/answer from the agent.
        messages: Full message history from the execution (for debugging/analysis).
        tool_calls_count: Total number of tool calls made during execution.
        iterations: Number of reasoning iterations performed.
        metadata: Additional metadata from the execution (e.g., timing, errors).

    Example:
        >>> result = AgentResult(
        ...     content="The answer is 42.",
        ...     tool_calls_count=3,
        ...     iterations=5,
        ...     metadata={"source": "react_agent"}
        ... )
        >>> print(f"Answer: {result.content}")
        >>> print(f"Tool calls: {result.tool_calls_count}")
    """

    content: str = Field(
        default="",
        description="The final text content/answer from the agent",
    )
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full message history from the execution",
    )
    tool_calls_count: int = Field(
        default=0,
        description="Total number of tool calls made during execution",
    )
    iterations: int = Field(
        default=0,
        description="Number of reasoning iterations performed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the execution",
    )


class BaseAgent(ABC):
    """Abstract base class for agents.

    An agent is responsible for running the reasoning loop. It receives initial
    messages and available tools, then iteratively reasons and acts until it
    produces a final answer.

    This abstraction allows different reasoning strategies to be plugged into
    the same AgenticGrader without changing the grader's code. Subclasses can
    implement various patterns like ReAct, Chain-of-Thought, Tree-of-Thought,
    or custom multi-agent systems.

    Attributes:
        model: The LLM model used for reasoning (BaseChatModel instance).
        tools: Dictionary mapping tool names to BaseTool instances.
        max_iterations: Maximum number of reasoning iterations to prevent infinite loops.
        callback: Optional callback function for processing model responses.

    Example:
        >>> class MyCustomAgent(BaseAgent):
        ...     async def arun(self, messages):
        ...         # Custom reasoning logic
        ...         return AgentResult(content="Final answer")
        >>>
        >>> # With model instance
        >>> agent = MyCustomAgent(model=my_model, tools=[tool1, tool2])
        >>>
        >>> # With dict config
        >>> agent = MyCustomAgent(
        ...     model={"model": "gpt-4", "api_key": "..."},
        ...     tools=[tool1, tool2],
        ... )
        >>> result = await agent.arun(messages)
    """

    def __init__(
        self,
        model: Union[BaseChatModel, Dict[str, Any]],
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        callback: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the agent.

        Args:
            model: The LLM model for reasoning. Can be either:
                   - A BaseChatModel instance with `achat` method
                   - A dict config (will be converted to OpenAIChatModel)
            tools: List of available tools for the agent. Each tool should be
                  a BaseTool instance.
            max_iterations: Maximum number of reasoning iterations to prevent
                           infinite loops. Defaults to 10.
            callback: Optional callback function for processing model responses.
                     Can be used for logging, metrics collection, etc.
        """
        if isinstance(model, dict):
            self.model: BaseChatModel = OpenAIChatModel(**model)
        else:
            self.model = model
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations
        self.callback = callback

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent.

        Args:
            tool: The tool to register. If a tool with the same name already
                 exists, it will be replaced.
        """
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool if found, None otherwise.
        """
        return self.tools.get(name)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schema for all registered tools.

        Returns:
            List of tool schemas in OpenAI function calling format.
        """
        return [tool.schema for tool in self.tools.values()]

    @abstractmethod
    async def arun(
        self,
        messages: List[Dict[str, Any]],
    ) -> AgentResult:
        """Run the agent reasoning loop asynchronously.

        This is the main method that subclasses must implement. It should take
        initial messages, run the reasoning loop (potentially calling tools),
        and return the final result.

        Args:
            messages: Initial messages in OpenAI format. Typically includes
                     a system message and user message with the task.

        Returns:
            AgentResult: The result containing the final answer and execution
                        metadata (tool calls count, iterations, etc.).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

    def run(self, messages: List[Dict[str, Any]]) -> AgentResult:
        """Run the agent reasoning loop synchronously.

        This is a convenience method for cases where async execution is not
        needed. It runs the async `arun` method in a new event loop.

        Args:
            messages: Initial messages in OpenAI format.

        Returns:
            AgentResult: The execution result.
        """
        return asyncio.run(self.arun(messages))


class ReActAgent(BaseAgent):
    """Built-in ReAct agent using OpenAI function calling.

    This agent implements the standard ReAct (Reasoning + Acting) pattern:
    1. LLM receives task + available tools
    2. LLM returns tool_calls (or final answer)
    3. Execute tools, append results to messages
    4. Repeat until LLM returns content without tool_calls

    This is OpenJudge's default agent implementation with zero external
    dependencies. It provides full control over the reasoning process and
    is optimized for evaluation tasks.

    Attributes:
        model: The LLM model for reasoning (BaseChatModel instance).
        tools: Dictionary mapping tool names to BaseTool instances.
        max_iterations: Maximum reasoning iterations.
        callback: Optional callback for model responses.
        truncate_tool_output: Maximum length for tool output before truncation.

    Example:
        >>> # With model instance
        >>> agent = ReActAgent(
        ...     model=OpenAIChatModel(model="gpt-4"),
        ...     tools=[WebSearchTool()],
        ... )
        >>>
        >>> # With dict config (convenience)
        >>> agent = ReActAgent(
        ...     model={"model": "gpt-4", "api_key": "..."},
        ...     tools=[WebSearchTool()],
        ...     max_iterations=10,
        ... )
        >>>
        >>> result = await agent.arun(messages)
        >>> print(result.content)
    """

    def __init__(
        self,
        model: Union[BaseChatModel, Dict[str, Any]],
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        callback: Optional[Callable[..., Any]] = None,
        truncate_tool_output: int = 4000,
    ):
        """Initialize the ReActAgent.

        Args:
            model: The LLM model for reasoning. Can be either:
                   - A BaseChatModel instance with `achat` method
                   - A dict config (will be converted to OpenAIChatModel)
            tools: List of available tools for the agent.
            max_iterations: Maximum number of reasoning iterations. Defaults to 10.
            callback: Optional callback for processing model responses.
            truncate_tool_output: Maximum length for tool output before truncation.
                                 Set to 0 to disable truncation. Defaults to 4000.
        """
        super().__init__(
            model=model,
            tools=tools,
            max_iterations=max_iterations,
            callback=callback,
        )
        self.truncate_tool_output = truncate_tool_output

    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool call and return the result message.

        Args:
            tool_call: OpenAI tool_call object containing:
                      - id: Unique identifier for the tool call
                      - function.name: Name of the tool to call
                      - function.arguments: JSON string of arguments

        Returns:
            Tool result message in OpenAI format with role="tool".
        """
        tool_call_id = tool_call.get("id", str(uuid.uuid4()))
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        # Parse arguments from JSON string
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {}

        # Get and execute tool
        tool = self.get_tool(tool_name)
        if not tool:
            content = f"Error: Tool '{tool_name}' not found. " f"Available tools: {list(self.tools.keys())}"
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

        # Truncate output if needed to prevent context overflow
        if self.truncate_tool_output > 0 and len(content) > self.truncate_tool_output:
            content = content[: self.truncate_tool_output] + "\n... [truncated]"

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def _normalize_tool_call(
        self,
        tool_call: Union[Dict[str, Any], Any],
    ) -> Dict[str, Any]:
        """Normalize a tool call to dictionary format.

        Handles both dictionary and object-style tool calls from different
        LLM providers.

        Args:
            tool_call: Tool call in either dict format or object format.

        Returns:
            Normalized tool call dictionary.
        """
        if isinstance(tool_call, dict):
            return {
                "id": tool_call.get("id", str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                },
            }
        else:
            # Handle object-style tool calls (e.g., from OpenAI SDK)
            return {
                "id": getattr(tool_call, "id", str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": getattr(tool_call.function, "name", ""),
                    "arguments": getattr(tool_call.function, "arguments", "{}"),
                },
            }

    async def arun(
        self,
        messages: List[Dict[str, Any]],
    ) -> AgentResult:
        """Run the ReAct loop until LLM produces final answer.

        The loop continues until either:
        1. The LLM returns content without tool_calls (final answer)
        2. Maximum iterations is reached

        Args:
            messages: Initial messages in OpenAI format (system + user).

        Returns:
            AgentResult containing the final answer and execution metadata.

        Raises:
            Exception: If the LLM call fails (propagated from model.achat).
        """
        # Make a copy to avoid modifying the original
        messages = list(messages)
        tools_schema = self.get_tools_schema()

        iteration = 0
        content = ""
        tool_calls_count = 0

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

            # Handle streaming response - consume all chunks
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
                # Normalize tool_calls to serializable format
                assistant_msg["tool_calls"] = [self._normalize_tool_call(tc) for tc in tool_calls]

            messages.append(assistant_msg)

            # If no tool calls, we have the final answer
            if not tool_calls:
                logger.debug(f"ReAct completed after {iteration} iterations")
                return AgentResult(
                    content=content,
                    messages=messages,
                    tool_calls_count=tool_calls_count,
                    iterations=iteration,
                )

            # Execute all tool calls
            for tc in assistant_msg.get("tool_calls", []):
                tool_result_msg = await self._execute_tool_call(tc)
                messages.append(tool_result_msg)
                tool_calls_count += 1

                # Log tool call details for debugging
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", "{}")
                tool_output = tool_result_msg["content"]
                logger.debug(f"Tool call: {tool_name}\n" f"  Args: {tool_args}\n" f"  Result: {tool_output[:200]}...")

        # Max iterations reached
        logger.warning(f"ReAct reached max iterations ({self.max_iterations})")
        return AgentResult(
            content=content,
            messages=messages,
            tool_calls_count=tool_calls_count,
            iterations=iteration,
            metadata={"max_iterations_reached": True},
        )
