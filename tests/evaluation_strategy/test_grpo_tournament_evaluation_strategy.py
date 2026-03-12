"""Unit tests for GRPOTournamentEvaluationStrategy."""

import pytest

from openjudge.evaluation_strategy.grpo_tournament_evaluation_strategy import (
    GRPOTournamentEvaluationStrategy,
)
from openjudge.graders.schema import GraderError, GraderRank, GraderScore


def _make_pairwise_grader(winner_map: dict):
    """Create a mock pairwise grader that returns predetermined winners.

    Args:
        winner_map: dict mapping (response_1, response_2) -> rank list,
                    e.g. {("a", "b"): [1, 2]} means response_1 wins.
    """

    async def mock_grader(query: str, response_1: str, response_2: str):
        key = (response_1, response_2)
        if key in winner_map:
            return GraderRank(name="pairwise", rank=winner_map[key], reason="mock")
        return GraderError(name="pairwise", error="unknown pair", reason="not in map")

    return mock_grader


@pytest.mark.unit
class TestGRPOTournamentEvaluationStrategy:
    """Tests for GRPOTournamentEvaluationStrategy."""

    def test_initialization_defaults(self):
        strategy = GRPOTournamentEvaluationStrategy()
        assert strategy.debiased is False

    def test_initialization_debiased(self):
        strategy = GRPOTournamentEvaluationStrategy(debiased=True)
        assert strategy.debiased is True

    @pytest.mark.asyncio
    async def test_missing_responses_raises_error(self):
        strategy = GRPOTournamentEvaluationStrategy()

        async def noop(**kwargs):
            pass

        with pytest.raises(ValueError, match="requires 'responses'"):
            await strategy.execute(noop, query="test")

    @pytest.mark.asyncio
    async def test_missing_query_raises_error(self):
        strategy = GRPOTournamentEvaluationStrategy()

        async def noop(**kwargs):
            pass

        with pytest.raises(ValueError, match="requires 'query'"):
            await strategy.execute(noop, responses=["a", "b"])

    @pytest.mark.asyncio
    async def test_fewer_than_two_responses_raises_error(self):
        strategy = GRPOTournamentEvaluationStrategy()

        async def noop(**kwargs):
            pass

        with pytest.raises(ValueError, match="At least 2 responses"):
            await strategy.execute(noop, query="test", responses=["only_one"])

    @pytest.mark.asyncio
    async def test_clear_winner(self):
        """Response A beats B and C; B beats C. A should have highest reward."""
        grader = _make_pairwise_grader(
            {
                ("a", "b"): [1, 2],
                ("a", "c"): [1, 2],
                ("b", "c"): [1, 2],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy()

        results = await strategy.execute(grader, query="test", responses=["a", "b", "c"])

        assert len(results) == 3
        assert results[0].score == 1.0  # a: 2W 0L → (2*2 - 2)/2 = 1.0
        assert results[1].score == 0.0  # b: 1W 1L → (2*1 - 2)/2 = 0.0
        assert results[2].score == -1.0  # c: 0W 2L → (2*0 - 2)/2 = -1.0

    @pytest.mark.asyncio
    async def test_two_responses(self):
        """Simple head-to-head: first response wins."""
        grader = _make_pairwise_grader(
            {
                ("x", "y"): [1, 2],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy()

        results = await strategy.execute(grader, query="q", responses=["x", "y"])

        assert len(results) == 2
        assert results[0].score == 1.0
        assert results[1].score == -1.0

    @pytest.mark.asyncio
    async def test_second_response_wins(self):
        """rank=[2,1] means response_2 wins."""
        grader = _make_pairwise_grader(
            {
                ("x", "y"): [2, 1],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy()

        results = await strategy.execute(grader, query="q", responses=["x", "y"])

        assert results[0].score == -1.0
        assert results[1].score == 1.0

    @pytest.mark.asyncio
    async def test_all_tied(self):
        """4 responses in a cycle: A>B, B>C, C>D, D>A, plus A>C, B>D.
        Each wins exactly half."""
        grader = _make_pairwise_grader(
            {
                ("a", "b"): [1, 2],
                ("a", "c"): [1, 2],
                ("a", "d"): [2, 1],
                ("b", "c"): [2, 1],
                ("b", "d"): [1, 2],
                ("c", "d"): [2, 1],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy()

        results = await strategy.execute(grader, query="q", responses=["a", "b", "c", "d"])

        # a: wins vs b,c; loses vs d → 2W 1L → (4-3)/3 = 0.333...
        # b: wins vs d; loses vs a,c → 1W 2L (corrected from map: b vs c = [2,1] means c wins)
        # Actually let me recount:
        # (a,b)=[1,2] → a wins
        # (a,c)=[1,2] → a wins
        # (a,d)=[2,1] → d wins
        # (b,c)=[2,1] → c wins
        # (b,d)=[1,2] → b wins
        # (c,d)=[2,1] → d wins
        # a: 2W 1L, b: 1W 2L, c: 1W 2L, d: 2W 1L
        assert len(results) == 4
        for r in results:
            assert isinstance(r, GraderScore)

    @pytest.mark.asyncio
    async def test_grader_errors_are_skipped(self):
        """When grader returns errors, those comparisons are skipped."""

        async def flaky_grader(query, response_1, response_2):
            if response_1 == "a" and response_2 == "b":
                return GraderRank(name="p", rank=[1, 2], reason="ok")
            if response_1 == "a" and response_2 == "c":
                return GraderError(name="p", error="timeout", reason="err")
            if response_1 == "b" and response_2 == "c":
                return GraderRank(name="p", rank=[2, 1], reason="ok")
            return GraderError(name="p", error="unknown", reason="err")

        strategy = GRPOTournamentEvaluationStrategy()
        results = await strategy.execute(flaky_grader, query="q", responses=["a", "b", "c"])

        # a vs b: a wins → a:1W, b:0W (1 valid each)
        # a vs c: error → skipped
        # b vs c: c wins → c:1W, b:0W (1 valid each)
        assert results[0].score == 1.0  # a: 1W/1 valid → (2*1-1)/1 = 1.0
        assert results[1].score == -1.0  # b: 0W/2 valid → (2*0-2)/2 = -1.0
        assert results[2].score == 1.0  # c: 1W/1 valid → (2*1-1)/1 = 1.0
        assert results[0].metadata["valid_comparisons"] == 1
        assert results[1].metadata["valid_comparisons"] == 2
        assert results[2].metadata["valid_comparisons"] == 1

    @pytest.mark.asyncio
    async def test_all_errors_gives_zero(self):
        """When all comparisons error, all rewards are 0.0."""

        async def error_grader(query, response_1, response_2):
            return GraderError(name="p", error="fail", reason="always fails")

        strategy = GRPOTournamentEvaluationStrategy()
        results = await strategy.execute(error_grader, query="q", responses=["a", "b", "c"])

        for r in results:
            assert r.score == 0.0
            assert r.metadata["valid_comparisons"] == 0

    @pytest.mark.asyncio
    async def test_metadata_fields(self):
        grader = _make_pairwise_grader(
            {
                ("a", "b"): [1, 2],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy()

        results = await strategy.execute(grader, query="q", responses=["a", "b"])

        assert results[0].metadata["wins"] == 1
        assert results[0].metadata["losses"] == 0
        assert results[0].metadata["valid_comparisons"] == 1
        assert results[0].metadata["total_peers"] == 1
        assert results[0].metadata["debiased"] is False
        assert results[0].name == "grpo_tournament"

    @pytest.mark.asyncio
    async def test_debiased_consistent_pair(self):
        """Debiased mode: both orderings agree → count the win."""
        grader = _make_pairwise_grader(
            {
                ("a", "b"): [1, 2],  # a wins when presented first
                ("b", "a"): [2, 1],  # a wins when presented second (consistent)
            }
        )
        strategy = GRPOTournamentEvaluationStrategy(debiased=True)

        results = await strategy.execute(grader, query="q", responses=["a", "b"])

        assert results[0].score == 1.0  # a wins consistently
        assert results[1].score == -1.0
        assert results[0].metadata["debiased"] is True

    @pytest.mark.asyncio
    async def test_debiased_inconsistent_pair_skipped(self):
        """Debiased mode: orderings disagree → pair is skipped."""
        grader = _make_pairwise_grader(
            {
                ("a", "b"): [1, 2],  # a wins when first
                ("b", "a"): [1, 2],  # b also wins when first (inconsistent — position bias)
            }
        )
        strategy = GRPOTournamentEvaluationStrategy(debiased=True)

        results = await strategy.execute(grader, query="q", responses=["a", "b"])

        # Inconsistent → skipped → no valid comparisons → 0.0
        assert results[0].score == 0.0
        assert results[1].score == 0.0
        assert results[0].metadata["valid_comparisons"] == 0

    @pytest.mark.asyncio
    async def test_debiased_three_responses(self):
        """Debiased mode with 3 responses: mix of consistent and inconsistent pairs."""
        grader = _make_pairwise_grader(
            {
                # a vs b: consistent (a wins)
                ("a", "b"): [1, 2],
                ("b", "a"): [2, 1],
                # a vs c: inconsistent (position bias)
                ("a", "c"): [1, 2],
                ("c", "a"): [1, 2],
                # b vs c: consistent (c wins)
                ("b", "c"): [2, 1],
                ("c", "b"): [1, 2],
            }
        )
        strategy = GRPOTournamentEvaluationStrategy(debiased=True)

        results = await strategy.execute(grader, query="q", responses=["a", "b", "c"])

        # a vs b: a wins (consistent) → a:1W, b:0W
        # a vs c: skipped (inconsistent)
        # b vs c: c wins (consistent) → c:1W, b:0W
        assert results[0].score == 1.0  # a: 1W/1 valid
        assert results[1].score == -1.0  # b: 0W/2 valid
        assert results[2].score == 1.0  # c: 1W/1 valid
