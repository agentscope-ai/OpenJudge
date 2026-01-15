# -*- coding: utf-8 -*-
"""Local execution controller implementation.

This module provides a local execution controller that manages concurrency
using a semaphore to limit the number of simultaneous operations.
"""

import asyncio
from typing import Any, Awaitable, Callable

from .base import BaseController, R


class LocalController(BaseController):
    """Local controller implementing resource management for local execution.

    This controller uses a Semaphore to limit the number of concurrent operations
    running locally, preventing resource exhaustion when executing many tasks.

    Examples:
        Basic usage:

        >>> controller = LocalController(max_concurrency=5)
        >>> async with controller:
        ...     result = await controller.submit(async_function, arg1="value1")
    """

    def __init__(self, max_concurrency: int = 32):
        """Initialize the local execution controller.

        Args:
            max_concurrency: Maximum number of concurrent operations allowed.
                           Defaults to 32.
        """
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def submit(self, fn: Callable[..., Awaitable[R]], **kwargs: Any) -> R:
        """Submit a task for local execution with concurrency control.

        This method wraps the provided function with concurrency control using
        the ConcurrencyManager. It ensures that no more than max_concurrency
        operations are running simultaneously.

        Args:
            fn: The asynchronous function to execute
            **kwargs: Arguments to pass to the function

        Returns:
            R: The result of the function execution, with type determined by
               the generic type parameter R

        Raises:
            Exception: Any exceptions raised by the function being executed
        """
        async with self._semaphore:
            return await fn(**kwargs)
