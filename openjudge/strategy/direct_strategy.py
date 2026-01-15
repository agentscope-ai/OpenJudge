from typing import Any, Awaitable, Callable

from openjudge.strategy.base import BaseStrategy


class DirectStrategy(BaseStrategy):
    """Direct execution strategy: executes evaluation once and returns the result.

    This strategy performs a single evaluation call and returns its result directly.
    It serves as the default strategy and maintains backward compatibility with
    the original behavior of graders.

    Examples:
        Basic usage:

        >>> strategy = DirectStrategy()
        >>> result = await strategy.execute(call_function, param="value")
    """

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function once directly.

        This method calls the provided function once and returns its result.

        Args:
            call_fn: Async function that submits the task to a controller
            **kwargs: Arguments for the evaluation

        Returns:
            Any: The result of a single evaluation call
        """
        return await call_fn(**kwargs)
