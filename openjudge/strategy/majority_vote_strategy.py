import asyncio
from typing import Any, Awaitable, Callable

from openjudge.strategy.base import BaseStrategy


class MajorityVoteStrategy(BaseStrategy):
    """Majority vote strategy: executes evaluation multiple times and aggregates results.

    This strategy executes the evaluation function multiple times and would typically
    aggregate the results based on a majority vote or similar mechanism. For now,
    it returns the first result as a placeholder until a full aggregation mechanism
    is implemented.

    Examples:
        Basic usage with 3 evaluations:

        >>> strategy = MajorityVoteStrategy(n=3)
        >>> result = await strategy.execute(call_function, param="value")
    """

    def __init__(self, n: int = 3):
        """Initialize the majority vote strategy.

        Args:
            n: Number of times to execute the evaluation (should be odd to avoid ties).
               Defaults to 3.

        Raises:
            ValueError: If n is not a positive integer
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        self.n = n

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function n times and aggregate results.

        This method submits n tasks concurrently and awaits all of them.
        Currently, it returns the first result as a placeholder; in a real
        implementation, this would aggregate the results appropriately.

        Args:
            call_fn: Async function that submits the task to a controller
            **kwargs: Arguments for the evaluation

        Returns:
            Any: Aggregated result based on majority vote of n executions.
                 Currently returns the first result as a placeholder.
        """
        # Submit n tasks concurrently
        tasks = [call_fn(**kwargs) for _ in range(self.n)]
        results = await asyncio.gather(*tasks)

        # For now, return the first result as a placeholder
        # In a real implementation, this would aggregate the results appropriately
        return results[0]
