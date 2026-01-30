# -*- coding: utf-8 -*-
"""Local execution resource implementation.

This module provides a local execution resource that manages concurrency
using a semaphore to limit the number of simultaneous operations.
"""

import asyncio
from typing import Any, Awaitable, Callable

from .base_resource_executor import BaseResourceExecutor, R


class SemaphoreResourceExecutor(BaseResourceExecutor):
    """Local resource implementing resource management for local execution.

    This resource uses a Semaphore to limit the number of concurrent operations
    running locally, preventing resource exhaustion when executing many tasks.

    Examples:
        Basic usage:

        >>> async def async_function(arg1: str) -> str:
        ...     return f"processed {arg1}"
        >>> executor = SemaphoreResourceExecutor(max_concurrency=5)
        >>> result = await executor.submit(async_function, arg1="value1")
    """

    def __init__(self, max_concurrency: int = 32):
        """Initialize the local execution resource.

        Args:
            max_concurrency: Maximum number of concurrent operations allowed.
                           Defaults to 32.
        """
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def submit(self, fn: Callable[..., Awaitable[R]], **kwargs: Any) -> R:
        """Submit a task for local execution with concurrency control.

        This method wraps the provided function with concurrency control using
        an asyncio.Semaphore. It ensures that no more than max_concurrency
        operations are running simultaneously.

        Args:
            fn: The asynchronous function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            R: The result of the function execution, with type determined by
               the generic type parameter R

        Raises:
            Exception: Any exceptions raised by the function being executed
        """
        async with self._semaphore:
            return await fn(**kwargs)
