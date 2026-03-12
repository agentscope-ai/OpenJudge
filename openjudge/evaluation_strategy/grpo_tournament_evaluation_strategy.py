"""GRPO tournament evaluation strategy: scores rollouts via all-pairs pairwise comparison."""

# -*- coding: utf-8 -*-

import asyncio
from itertools import combinations
from typing import Any, Awaitable, Callable, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.schema import GraderRank, GraderScore


class GRPOTournamentEvaluationStrategy(BaseEvaluationStrategy):
    """GRPO tournament strategy: compares every pair of rollouts and returns net win rate.

    For a GRPO group of N rollouts to the same prompt, this strategy runs all
    N*(N-1)/2 pairwise comparisons using a LISTWISE grader and computes each
    rollout's net win rate as its reward signal.

    The reward for rollout i is:
        r_i = (wins_i - losses_i) / (N - 1)

    This produces rewards in [-1.0, 1.0] that are relative within the group,
    robust to absolute scale drift, and well-suited for subjective tasks where
    LLM judges find relative comparison easier than absolute scoring.

    Attributes:
        debiased (bool): If True, each pair is compared in both orders (A,B) and
            (B,A); a win is counted only when both orderings agree. Inconsistent
            pairs are skipped. This doubles the number of LLM calls but reduces
            position bias.

    Examples:
        >>> strategy = GRPOTournamentEvaluationStrategy()
        >>> rewards = await strategy.execute(
        ...     pairwise_grader.aevaluate,
        ...     query="Write a haiku about the ocean.",
        ...     responses=["response_1", "response_2", "response_3"],
        ... )
        >>> # Returns list[GraderScore] with net win rates
    """

    def __init__(self, debiased: bool = False):
        """Initialize the GRPO tournament strategy.

        Args:
            debiased (bool): Whether to run double-sided comparison to mitigate
                position bias (default False). When True, the number of LLM calls
                doubles from N*(N-1)/2 to N*(N-1).
        """
        self.debiased = debiased

    async def _compare_pair(
        self,
        call_fn: Callable[..., Awaitable[Any]],
        query: str,
        resp_i: str,
        resp_j: str,
    ) -> int | None:
        """Compare a single pair and return the index of the winner.

        Returns:
            0 if resp_i wins, 1 if resp_j wins, None on error.
        """
        result = await call_fn(query=query, response_1=resp_i, response_2=resp_j)
        if not isinstance(result, GraderRank):
            return None
        return 0 if result.rank[0] == 1 else 1

    async def _compare_pair_debiased(
        self,
        call_fn: Callable[..., Awaitable[Any]],
        query: str,
        resp_i: str,
        resp_j: str,
    ) -> int | None:
        """Compare a pair in both orders; return winner only if both agree.

        Returns:
            0 if resp_i wins (consistent), 1 if resp_j wins (consistent),
            None if results are inconsistent or errored.
        """
        ab, ba = await asyncio.gather(
            call_fn(query=query, response_1=resp_i, response_2=resp_j),
            call_fn(query=query, response_1=resp_j, response_2=resp_i),
        )
        if not isinstance(ab, GraderRank) or not isinstance(ba, GraderRank):
            return None

        i_wins_ab = ab.rank[0] == 1
        j_wins_ba = ba.rank[0] == 1

        if i_wins_ab and not j_wins_ba:
            return 0
        elif not i_wins_ab and j_wins_ba:
            return 1
        return None

    async def execute(self, call_fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        """Run all-pairs tournament and return net win rate rewards.

        Args:
            call_fn: An async pairwise grader function (LISTWISE mode with 2 candidates).
                Must accept ``query``, ``response_1``, ``response_2`` keyword arguments
                and return a ``GraderRank``.
            **kwargs: Must include:
                - ``query`` (str): The prompt all rollouts respond to.
                - ``responses`` (list[str]): The group of rollout responses to compare.

        Returns:
            list[GraderScore]: One score per rollout. ``score`` is the net win rate
            in [-1.0, 1.0].

        Raises:
            ValueError: If fewer than 2 responses are provided.
            ValueError: If ``responses`` is not in kwargs.
        """
        if "responses" not in kwargs:
            raise ValueError("GRPOTournamentEvaluationStrategy requires 'responses' (list[str]) in kwargs.")
        if "query" not in kwargs:
            raise ValueError("GRPOTournamentEvaluationStrategy requires 'query' (str) in kwargs.")

        query: str = kwargs["query"]
        responses: List[str] = kwargs["responses"]
        n = len(responses)

        if n < 2:
            raise ValueError("At least 2 responses are required for a tournament.")

        pairs = list(combinations(range(n), 2))
        compare_fn = self._compare_pair_debiased if self.debiased else self._compare_pair

        outcomes = await asyncio.gather(*[compare_fn(call_fn, query, responses[i], responses[j]) for i, j in pairs])

        wins = [0] * n
        valid_comparisons = [0] * n

        for (i, j), outcome in zip(pairs, outcomes):
            if outcome is None:
                continue
            valid_comparisons[i] += 1
            valid_comparisons[j] += 1
            if outcome == 0:
                wins[i] += 1
            else:
                wins[j] += 1

        results: List[GraderScore] = []
        for idx in range(n):
            c = valid_comparisons[idx]
            if c > 0:
                net_win_rate = (2 * wins[idx] - c) / c
            else:
                net_win_rate = 0.0

            results.append(
                GraderScore(
                    name="grpo_tournament",
                    score=net_win_rate,
                    reason=f"Net win rate: {wins[idx]}W / {c - wins[idx]}L " f"out of {c} valid comparison(s).",
                    metadata={
                        "wins": wins[idx],
                        "losses": c - wins[idx],
                        "valid_comparisons": c,
                        "total_peers": n - 1,
                        "debiased": self.debiased,
                    },
                )
            )

        return results
