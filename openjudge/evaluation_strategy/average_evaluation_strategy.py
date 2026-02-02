"""Average evaluation strategy: aggregates results by calculating the mean value."""

# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Awaitable, Callable, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.schema import GraderScore


class AverageEvaluationStrategy(BaseEvaluationStrategy):
    """Average evaluation strategy: executes the evaluation function multiple times and returns the average result.

    This strategy runs the evaluation function multiple times (specified by num_evaluations)
    and aggregates numerical results by computing their average. It's useful for reducing
    noise in evaluations that return continuous numerical values.

    Attributes:
        num_evaluations (int): Number of times to execute the evaluation function.

    Examples:
        >>> strategy = AverageEvaluationStrategy(num_evaluations=5)
        >>> result = await strategy.execute(eval_fn, input_data="test")
        >>> # Executes eval_fn(input_data="test") 5 times and returns the average score
    """

    def __init__(self, num_evaluations: int = 3):
        """Initialize the average strategy.

        Args:
            num_evaluations (int): Number of evaluations to average (default 3).
        """
        if num_evaluations < 2:
            raise ValueError("num_evaluations must be at least 2")

        self.num_evaluations = num_evaluations

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function multiple times and return the average result.

        For GraderScore results, computes the average of the score values.
        For GraderRank results, this strategy may not be appropriate since ranks are not numerical.

        Args:
            call_fn: An async function that performs the evaluation.
                     Calling await call_fn(**kwargs) executes the evaluation.
            **kwargs: Arguments passed to the evaluation function.

        Returns:
            Any: The averaged result from all executions.
        """
        results: List[Any] = []
        coroutines = []

        # Execute the evaluation function multiple times
        for _ in range(self.num_evaluations):
            coroutines.append(call_fn(**kwargs))

        results = await asyncio.gather(*coroutines)

        # Handle empty results
        if not results:
            raise ValueError("No results returned from evaluation function.")

        # Filter valid results and determine result type
        valid_score_results = [r for r in results if isinstance(r, GraderScore)]

        # Process GraderScore results
        if valid_score_results:
            # Calculate the average of the scores
            avg_score_value = sum(r.score for r in valid_score_results) / len(valid_score_results)

            # Take the first valid result as a template and update its score
            first_result = valid_score_results[0]
            return GraderScore(
                name=first_result.name,
                score=avg_score_value,
                reason=f"Averaged from {len(valid_score_results)} valid evaluations "
                f"out of {self.num_evaluations} total.",
                metadata=getattr(first_result, "metadata", {}),
            )
        else:
            raise ValueError("No valid GraderScore results were returned from evaluation function.")
