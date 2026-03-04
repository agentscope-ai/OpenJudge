"""Direct evaluation strategy: runs evaluation once without any aggregation."""

# -*- coding: utf-8 -*-

from typing import Any, Awaitable, Callable

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)


class DirectEvaluationStrategy(BaseEvaluationStrategy):
    """Direct evaluation strategy: executes the evaluation function once and returns the result.

    This is the default strategy that simply forwards the call to the evaluation function
    without any additional processing or result aggregation. It serves as the baseline
    for other more complex strategies.

    Examples:
        >>> strategy = DirectEvaluationStrategy()
        >>> result = await strategy.execute(call_fn, input_data="test")
        >>> # Executes call_fn(input_data="test") once and returns its result
    """

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function once and return its result directly.

        Args:
            call_fn: An async function that performs the evaluation.
                     Calling await call_fn(**kwargs) executes the evaluation.
            **kwargs: Arguments passed to the evaluation function.

        Returns:
            Any: The direct result from the evaluation function.
        """
        return await call_fn(**kwargs)
