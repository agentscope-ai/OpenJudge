"""Voting evaluation strategy: aggregates results by selecting the most frequent outcome."""

# -*- coding: utf-8 -*-

import asyncio
from collections import Counter
from typing import Any, Awaitable, Callable, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.schema import GraderScore


class VotingEvaluationStrategy(BaseEvaluationStrategy):
    """Voting evaluation strategy: executes the evaluation function multiple times and returns the most frequent result.

    This strategy runs the evaluation function multiple times (specified by num_votes)
    and aggregates the results by selecting the most frequently occurring outcome.
    It's particularly useful for reducing noise in stochastic evaluations.

    Tips:
        To avoid ties, consider using an odd number for num_votes. For example:
        - 3, 5, 7... votes reduce chance of ties in binary outcomes
        - In case of a tie, the lowest score will be selected by default

    Attributes:
        num_votes (int): Number of times to execute the evaluation function.

    Examples:
        >>> strategy = VotingEvaluationStrategy(num_votes=5)
        >>> result = await strategy.execute(call_fn, input_data="test")
        >>> # Executes call_fn(input_data="test") 5 times and returns the most common result
    """

    def __init__(self, num_votes: int = 3):
        """Initialize the voting strategy.

        Args:
            num_votes (int): Number of votes/repetitions (default 3).
                             Using odd numbers can help avoid ties.
        """
        if num_votes < 2:
            raise ValueError("num_votes must be at least 2")

        self.num_votes = num_votes

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function multiple times and return the most frequent result.

        Args:
            call_fn: An async function that performs the evaluation.
                     Calling await call_fn(**kwargs) executes the evaluation.
            **kwargs: Arguments passed to the evaluation function.

        Returns:
            Any: The most frequent result from all executions.
                 In case of a tie among most frequent results, returns the lowest score.
        """
        results: List[Any] = []
        coroutines = []

        # Execute the evaluation function multiple times
        for _ in range(self.num_votes):
            coroutines.append(call_fn(**kwargs))

        results = await asyncio.gather(*coroutines)

        values = [result.score for result in results if hasattr(result, "score")]
        if len(values) == 0:
            raise ValueError(
                "VotingEvaluationStrategy only supports GraderScore."
                "No results were returned from the evaluation correctly."
            )

        counter = Counter(values)

        # Get all items sorted by count (descending), and by score (ascending) for ties
        most_common_items = counter.most_common()

        # Find the highest frequency
        max_frequency = most_common_items[0][1]

        # Filter to get all items with the highest frequency
        highest_freq_values = [item[0] for item in most_common_items if item[1] == max_frequency]

        # If there's a tie among the most frequent items, select the lowest value
        if len(highest_freq_values) > 1:
            most_common_value = min(highest_freq_values)
        else:
            most_common_value = highest_freq_values[0]

        # TODO: Even with odd number of votes, there can still be cases where multiple
        # different scores appear with the same highest frequency. For example:
        # - With 5 votes: [1, 3, 5, 5, 3] would have both 3 and 5 appearing twice
        # - With 7 votes: [1, 1, 2, 2, 3, 3, 4] would have 1, 2, and 3 all appearing twice
        # Consider implementing more sophisticated tie-breaking mechanisms in the future

        name = ""
        for r in results:
            if hasattr(r, "name"):
                name = r.name
                break

        # Find the first result matching the most common value
        return GraderScore(
            name=name,
            score=most_common_value,
            reason=f"Vote from {self.num_votes} evaluations.",
            metadata={"original_results": results},
        )
