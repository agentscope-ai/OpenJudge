# -*- coding: utf-8 -*-
"""Utilities for managing concurrency in OpenJudge.

This module provides utilities for controlling concurrent execution of
evaluations to prevent resource exhaustion and manage system load.
"""

import asyncio


class ConcurrencyManager:
    """
    A manager for controlling concurrency across the entire system.

    This class provides a centralized way to manage and limit the concurrent execution
    of grader evaluations to prevent resource exhaustion.
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConcurrencyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_concurrency: int = 32):
        """Initialize the GraderConcurrencyManager as a singleton.

        The initialization logic is guarded so that concurrency settings are
        only applied on the first construction. Subsequent constructions will
        reuse the existing semaphore and configuration without resetting the
        global limit.
        """
        if hasattr(self, "_semaphore"):
            # Already initialized, do not override existing concurrency settings
            return

        self.set_max_concurrency(max_concurrency)

    def set_max_concurrency(self, max_concurrency: int = 32):
        """
        Set the maximum number of concurrent grader evaluations.

        Args:
            max_concurrency: Maximum number of concurrent evaluations allowed
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")

        self._max_concurrency = max_concurrency
        # Note: We cannot directly change the semaphore's capacity,
        # so we create a new one
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def get_max_concurrency(self) -> int:
        """
        Get the current maximum concurrent limit.

        Returns:
            The maximum number of concurrent evaluations allowed
        """
        return self._max_concurrency

    async def run_with_concurrency_control(self, coro):
        """
        Run a coroutine with concurrency control.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        async with self._semaphore:
            return await coro

    @property
    def current_semaphore(self):
        """
        Get the current semaphore being used for concurrency control.

        Returns:
            The asyncio.Semaphore instance
        """
        return self._semaphore
