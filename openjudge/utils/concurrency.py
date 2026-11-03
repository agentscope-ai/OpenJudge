# -*- coding: utf-8 -*-
"""Concurrency utilities for managing async task execution.

Provides a ConcurrencyManager that wraps asyncio.Semaphore to limit
the number of concurrent coroutines.
"""

import asyncio
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

T = TypeVar("T")


async def _gather_with_cleanup(*coroutines: Awaitable[T], on_complete: Callable[[], Any] | None = None) -> list[T]:
    """Gather owned operations in order, cancelling and draining siblings on failure.

    Args:
        *coroutines: Operations whose lifetime belongs to this call.
        on_complete: Optional callback invoked once for each completed operation.

    Returns:
        Results in the same order as the supplied operations.
    """
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    if on_complete is not None:
        for task in tasks:
            task.add_done_callback(lambda _: on_complete())
    group = asyncio.gather(*tasks)
    try:
        # This scope owns cancellation, including while children perform async cleanup.
        return await asyncio.shield(group)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        cleanup = asyncio.gather(group, *tasks, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        raise


class ConcurrencyManager:
    """Manage concurrency for async tasks using an asyncio.Semaphore.

    Example:
        >>> mgr = ConcurrencyManager()
        >>> mgr.set_max_concurrency(10)
        >>> result = await mgr.run_with_concurrency_control(some_coro())
    """

    def __init__(self, max_concurrency: int = 10) -> None:
        """Initialize the manager with a positive concurrency limit.

        Args:
            max_concurrency: Maximum number of concurrent tasks. Must be greater than 0.

        Raises:
            ValueError: If max_concurrency is not positive.
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def set_max_concurrency(self, max_concurrency: int) -> None:
        """Update the max concurrency limit.

        Args:
            max_concurrency: Maximum number of concurrent tasks. Must be greater than 0.

        Raises:
            ValueError: If max_concurrency is not positive. The existing limit is unchanged.
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")
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
