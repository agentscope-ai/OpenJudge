# -*- coding: utf-8 -*-
"""
Tool system for agentic evaluation.

This module provides the tool infrastructure for OpenJudge's agentic evaluation,
including the base tool class and standardized result format. Tools define what
capabilities an agent has (e.g., web search, code execution, file operations).

The tool system follows OpenAI's function calling format, which is the de-facto
standard for LLM tool calling and ensures compatibility with most LLM providers.

Classes:
    ToolResult: Standardized result format for tool execution.
    BaseTool: Abstract base class for all tools.

Example:
    >>> from openjudge.agentic import BaseTool, ToolResult
    >>>
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

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

__all__ = [
    "ToolResult",
    "BaseTool",
]


class ToolResult(BaseModel):
    """Standardized result format for tool execution.

    This class provides a consistent interface for all tool outputs in OpenJudge,
    regardless of the underlying tool implementation. It captures both successful
    results and error conditions in a structured format.

    Attributes:
        success: Whether the tool execution was successful.
        output: The output data from the tool execution. Can be any type.
        error: Error message if the execution failed, None otherwise.
        metadata: Additional metadata from the execution (e.g., timing, source).

    Example:
        >>> # Successful execution
        >>> result = ToolResult(success=True, output="Search results: ...")
        >>> if result.success:
        ...     print(result.output)
        >>>
        >>> # Failed execution
        >>> result = ToolResult(success=False, error="Connection timeout")
        >>> if not result.success:
        ...     print(f"Error: {result.error}")
    """

    success: bool = Field(
        default=True,
        description="Whether the tool execution was successful",
    )
    output: Any = Field(
        default=None,
        description="The output data from the tool execution",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the execution failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the execution",
    )


class BaseTool(ABC):
    """Abstract base class for tools using OpenAI function calling format.

    All tools in OpenJudge should inherit from this class and implement:
    1. The `schema` class attribute with the tool's JSON schema
    2. The `aexecute` method for async execution

    The schema follows OpenAI's function calling format, which provides a
    standardized way to describe tool interfaces that works with most LLM
    providers (OpenAI, Anthropic, etc.).

    Attributes:
        schema: OpenAI function calling schema defining the tool's interface.
               Must include "type", "function.name", "function.description",
               and "function.parameters".

    Example:
        >>> class CodeExecutionTool(BaseTool):
        ...     schema = {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "execute_code",
        ...             "description": "Execute Python code and return the result",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "code": {
        ...                         "type": "string",
        ...                         "description": "Python code to execute"
        ...                     }
        ...                 },
        ...                 "required": ["code"]
        ...             }
        ...         }
        ...     }
        ...
        ...     async def aexecute(self, code: str, **kwargs) -> ToolResult:
        ...         try:
        ...             result = exec(code)
        ...             return ToolResult(success=True, output=result)
        ...         except Exception as e:
        ...             return ToolResult(success=False, error=str(e))
    """

    # OpenAI function calling format - subclasses MUST override this
    schema: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "base_tool",
            "description": "Base tool - override this in subclasses",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    @property
    def name(self) -> str:
        """Get the tool name from schema.

        Returns:
            The tool name as defined in the schema, or "unknown" if not found.
        """
        return self.schema.get("function", {}).get("name", "unknown")

    @property
    def description(self) -> str:
        """Get the tool description from schema.

        Returns:
            The tool description as defined in the schema, or empty string if not found.
        """
        return self.schema.get("function", {}).get("description", "")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the tool parameters schema.

        Returns:
            The parameters JSON schema as defined in the schema.
        """
        return self.schema.get("function", {}).get("parameters", {})

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool asynchronously with given parameters.

        This is the main method that subclasses must implement. It should be
        async to support non-blocking I/O operations (e.g., network requests,
        file operations).

        Args:
            **kwargs: Tool-specific parameters as defined in the schema.
                     The parameter names and types should match the schema.

        Returns:
            ToolResult: The result of the tool execution, containing success
                       status, output data, and optional error/metadata.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool synchronously.

        This is a convenience method for cases where async execution is not
        needed. It runs the async `aexecute` method in a new event loop.

        Note:
            This method creates a new event loop for each call, which may have
            performance implications. For better performance in async contexts,
            use `aexecute` directly.

        Args:
            **kwargs: Tool-specific parameters as defined in the schema.

        Returns:
            ToolResult: The result of the tool execution.
        """
        return asyncio.run(self.aexecute(**kwargs))
