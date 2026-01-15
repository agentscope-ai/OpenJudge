# -*- coding: utf-8 -*-
"""Base class for evaluation strategies.

This module defines the abstract base class for evaluation strategies that determine
how evaluations are performed, including execution patterns, retries, and result aggregation.
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable


class BaseStrategy(ABC):
    """Base evaluation strategy class: defines the evaluation workflow.

    Strategies determine how many times an evaluation is called, in what order,
    and how results are aggregated. They are independent of execution environment
    (local, distributed, etc.).

    This is an abstract base class that defines the interface for all evaluation
    strategies. Subclasses must implement the execute method to define their
    specific evaluation logic.

    Examples:
        Basic usage:

        >>> class MyStrategy(BaseStrategy):
        ...     async def execute(self, call_fn, **kwargs):
        ...         return await call_fn(**kwargs)
        ...
        >>> strategy = MyStrategy()
        >>> result = await strategy.execute(call_function, param="value")
    """

    @abstractmethod
    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation strategy.

        This abstract method defines how the evaluation strategy executes.
        The call_fn parameter is a function that submits tasks to a controller,
        and calling await call_fn(**kwargs) will submit the task to the controller.

        Args:
            call_fn: An async function that submits tasks to a controller.
                     Calling await call_fn(**kwargs) will submit the task to the controller.
            **kwargs: Arguments for the evaluation

        Returns:
            Any: The result of the strategy's execution and aggregation.
                 The specific return type depends on the strategy implementation.

        Raises:
            Exception: Any exceptions raised during strategy execution
        """
