# -*- coding: utf-8 -*-
"""Concurrency utilities for managing async task execution.

Provides a ConcurrencyManager that wraps asyncio.Semaphore to limit
the number of concurrent coroutines.
"""

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class ConcurrencyManager:
    """Manage concurrency for async tasks using an asyncio.Semaphore.

    Example:
        >>> mgr = ConcurrencyManager()
        >>> mgr.set_max_concurrency(10)
        >>> result = await mgr.run_with_concurrency_control(some_coro())
    """

    def __init__(self, max_concurrency: int = 10):
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def set_max_concurrency(self, max_concurrency: int) -> None:
        """Update the max concurrency limit.

        Args:
            max_concurrency: Maximum number of concurrent tasks.
        """
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_concurrency_control(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine with concurrency control.

        Acquires the semaphore before executing the coroutine, ensuring
        that at most ``max_concurrency`` coroutines run simultaneously.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine.
        """
        async with self._semaphore:
            return await coro
