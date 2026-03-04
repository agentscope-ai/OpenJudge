# -*- coding: utf-8 -*-
"""
Function adapter for wrapping simple Python functions as tools.

This module provides the FunctionToolAdapter class, which allows you to create
OpenJudge tools from simple Python functions without needing to define a full
BaseTool subclass. This is useful for quick prototyping or wrapping existing
utility functions.

Classes:
    FunctionToolAdapter: Adapter for wrapping Python functions as BaseTool.

Example:
    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>>
    >>> def calculate_sum(a: int, b: int) -> int:
    ...     return a + b
    >>>
    >>> tool = FunctionToolAdapter(
    ...     func=calculate_sum,
    ...     name="calculate_sum",
    ...     description="Calculate the sum of two numbers",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {
    ...             "a": {"type": "integer", "description": "First number"},
    ...             "b": {"type": "integer", "description": "Second number"}
    ...         },
    ...         "required": ["a", "b"]
    ...     }
    ... )
    >>> result = await tool.aexecute(a=1, b=2)
    >>> print(result.output)  # 3
"""

import asyncio
from typing import Any, Callable, Dict, Optional

from loguru import logger

from ..tools import BaseTool, ToolResult

__all__ = [
    "FunctionToolAdapter",
]


class FunctionToolAdapter(BaseTool):
    """Adapter for wrapping a simple Python function as a BaseTool.

    This is a convenience class for creating tools from simple functions
    without needing to define a full BaseTool subclass. It supports both
    synchronous and asynchronous functions.

    Attributes:
        schema: OpenAI function calling schema built from the provided parameters.

    Example:
        >>> # Synchronous function
        >>> def search_web(query: str) -> str:
        ...     return "results..."
        >>>
        >>> tool = FunctionToolAdapter(
        ...     func=search_web,
        ...     name="web_search",
        ...     description="Search the web",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"query": {"type": "string"}},
        ...         "required": ["query"]
        ...     }
        ... )
        >>>
        >>> # Async function
        >>> async def fetch_data(url: str) -> str:
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.text()
        >>>
        >>> async_tool = FunctionToolAdapter(
        ...     func=fetch_data,
        ...     name="fetch_data",
        ...     description="Fetch data from a URL",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"url": {"type": "string"}},
        ...         "required": ["url"]
        ...     }
        ... )
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the FunctionToolAdapter.

        Args:
            func: The function to wrap. Can be synchronous or asynchronous.
            name: The tool name. Should be a valid identifier (alphanumeric + underscore).
            description: Human-readable description of what the tool does.
                        This is shown to the LLM to help it decide when to use the tool.
            parameters: Optional JSON schema for the function parameters.
                       If not provided, defaults to an empty object schema.
                       Should follow JSON Schema format with "type", "properties",
                       and "required" fields.
        """
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)

        # Build OpenAI function calling schema
        self.schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def aexecute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function asynchronously.

        Args:
            **kwargs: Function parameters. Should match the parameters schema.

        Returns:
            ToolResult: The execution result. On success, output contains the
                       function's return value. On failure, error contains the
                       exception message.
        """
        try:
            if self._is_async:
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            logger.warning(f"Function tool '{self.name}' execution failed: {e}")
            return ToolResult(success=False, error=str(e))
