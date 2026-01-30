"""Average evaluation strategy: aggregates results by calculating the mean value."""

# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Awaitable, Callable, List

import numpy as np

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.schema import GraderError, GraderRank, GraderScore


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

        # If we have GraderScore objects, compute the average score
        if results and isinstance(results[0], GraderScore):
            # Calculate the average of the scores
            avg_score_value = sum(getattr(r, "score", 0) for r in results) / len(results)

            # Take the first result as a template and update its score
            first_result = results[0]
            return GraderScore(
                name=first_result.name,
                score=avg_score_value,
                reason=f"Averaged from {self.num_evaluations} evaluations. Original: {first_result.reason}",
                metadata=getattr(first_result, "metadata", {}),
            )

        # For GraderRank objects, average strategy might not be ideal, but we can average the ranks if they are numbers
        elif results and isinstance(results[0], GraderRank):
            # This case is trickier since ranks are ordered lists
            # For now, we'll return the first result, but in the future we might want to implement
            # more sophisticated rank averaging algorithms
            valid_results = []
            for r in results:
                try:
                    valid_results.append(np.array(r.rank))
                except Exception:
                    continue

            if valid_results:
                # Calculate the average rank for each item position
                avg_rank = np.mean(valid_results, axis=0).tolist()

                # Create pairs of (item_index, average_rank_value)
                indexed_avg_ranks = [(i, avg_rank[i]) for i in range(len(avg_rank))]

                # Sort by the average rank value (ascending order - lower rank is better)
                sorted_by_avg_rank = sorted(indexed_avg_ranks, key=lambda x: x[1])

                # Create the new rank list where rank[i] represents the rank of the i-th item
                # in the original list. We assign ranks from 1 to n based on the average performance
                new_rank = [0] * len(avg_rank)
                for new_rank_idx, (original_idx, _) in enumerate(sorted_by_avg_rank):
                    new_rank[original_idx] = new_rank_idx + 1

                first_result = results[0]
                return GraderRank(
                    name=first_result.name,
                    rank=new_rank,
                    reason=f"Averaged from {self.num_evaluations} evaluations.",
                    metadata={
                        "original_results": results,
                    },
                )
            else:
                first_result = results[0]
                return GraderError(
                    name=first_result.name,
                    reason=f"Could not average ranks from {self.num_evaluations} evaluations.",
                    error="All ranks are not numerical.",
                    metadata={
                        "original_results": results,
                    },
                )
        else:
            raise ValueError("AverageEvaluationStrategy can only handle GraderScore or GraderRank results.")
