"""Voting evaluation strategy: aggregates results by selecting the most frequent outcome."""

# -*- coding: utf-8 -*-

import asyncio
from collections import Counter
from typing import Any, Awaitable, Callable, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.schema import GraderScore

SUPPORTED_TIE_BREAKERS = {"min", "max", "mean_closest"}


class VotingEvaluationStrategy(BaseEvaluationStrategy):
    """Voting evaluation strategy: executes the evaluation function multiple times and returns the most frequent result.

    This strategy runs the evaluation function multiple times (specified by num_votes)
    and aggregates the results by selecting the most frequently occurring outcome.
    It's particularly useful for reducing noise in stochastic evaluations.

    Tips:
        To avoid ties, consider using an odd number for num_votes. For example:
        - 3, 5, 7... votes reduce chance of ties in binary outcomes
        - In case of a tie, tie_breaker controls which score is selected

    Attributes:
        num_votes (int): Number of times to execute the evaluation function.
        tie_breaker (str | Callable): Tie-breaking strategy when top vote counts are tied.

    Examples:
        >>> strategy = VotingEvaluationStrategy(num_votes=5)
        >>> result = await strategy.execute(call_fn, input_data="test")
        >>> # Executes call_fn(input_data="test") 5 times and returns the most common result
    """

    def __init__(
        self,
        num_votes: int = 3,
        tie_breaker: str | Callable[[list[float], list[float]], float] = "min",
    ):
        """Initialize the voting strategy.

        Args:
            num_votes (int): Number of votes/repetitions (default 3).
                             Using odd numbers can help avoid ties.
            tie_breaker (str | Callable[[list[float], list[float]], float]):
                Tie-breaking strategy when highest vote counts are tied.
                Supported string values:
                - "min": select the lowest tied score
                - "max": select the highest tied score
                - "mean_closest": select tied score closest to mean of all scores
                You can also pass a callable receiving
                (tie_candidates, all_scores) and returning one tie candidate.
        """
        if num_votes < 2:
            raise ValueError("num_votes must be at least 2")

        is_valid_str_tie_breaker = isinstance(tie_breaker, str) and tie_breaker in SUPPORTED_TIE_BREAKERS
        if not (is_valid_str_tie_breaker or callable(tie_breaker)):
            raise ValueError(f"tie_breaker must be one of {SUPPORTED_TIE_BREAKERS} or a callable")

        self.num_votes = num_votes
        self.tie_breaker = tie_breaker

    def _resolve_tie(self, tie_candidates: list[float], all_scores: list[float]) -> float:
        """Resolve tie among candidates using configured tie_breaker."""
        if len(tie_candidates) == 1:
            return tie_candidates[0]

        if callable(self.tie_breaker):
            selected = self.tie_breaker(tie_candidates, all_scores)
            if selected not in tie_candidates:
                raise ValueError(
                    "Custom tie_breaker must return one of the tie candidates. "
                    f"Got: {selected}, candidates: {tie_candidates}"
                )
            return selected

        if self.tie_breaker == "min":
            return min(tie_candidates)
        if self.tie_breaker == "max":
            return max(tie_candidates)
        if self.tie_breaker == "mean_closest":
            mean_score = sum(all_scores) / len(all_scores)
            return min(tie_candidates, key=lambda score: (abs(score - mean_score), score))

        raise ValueError(f"Unsupported tie_breaker: {self.tie_breaker}")

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Execute the evaluation function multiple times and return the most frequent result.

        Args:
            call_fn: An async function that performs the evaluation.
                     Calling await call_fn(**kwargs) executes the evaluation.
            **kwargs: Arguments passed to the evaluation function.

        Returns:
            Any: The most frequent result from all executions.
                 In case of a tie among most frequent results, tie_breaker resolves it.
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

        most_common_value = self._resolve_tie(highest_freq_values, values)

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
