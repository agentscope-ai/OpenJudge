# -*- coding: utf-8 -*-
"""Base class for execution controllers.

This module defines the abstract base class for execution controllers that manage
how tasks are submitted and executed, providing resource management and concurrency control.
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, TypeVar

R = TypeVar("R")


class BaseController(ABC):
    """Resource controller base class: defines task submission interface.

    Execution controllers manage how tasks are submitted and executed, handling
    resource allocation, concurrency control, and potentially distributed execution.

    This is an abstract base class that defines the interface for all execution
    controllers. Subclasses must implement the submit method to define how tasks
    are executed in their specific environment.

    Examples:
        Basic usage with async context manager:

        ... result = await controller.submit(my_async_func, param="value")
    """

    @abstractmethod
    async def submit(self, fn: Callable[..., Awaitable[R]], **kwargs: Any) -> R:
        """Submit a task for execution.

        This abstract method defines how tasks are submitted for execution.
        Implementations should handle the actual execution of the function
        in their specific environment (local, distributed, etc.) and apply
        any necessary resource management or concurrency control.

        Args:
            fn: An asynchronous function to execute (typically grader.aevaluate)
            **kwargs: Arguments to pass to the function

        Returns:
            R: The result of the function execution, with type determined by
               the generic type parameter R

        Raises:
            Exception: Any exceptions raised by the function being executed
        """
