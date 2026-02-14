"""Unit tests for VotingEvaluationStrategy.

This module contains tests for VotingEvaluationStrategy to ensure it functions
correctly according to the specifications.
"""

import pytest

from openjudge.evaluation_strategy.voting_evaluation_strategy import (
    CLOSEST_TO_MEAN,
    MAX,
    MIN,
    VotingEvaluationStrategy,
)
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestVotingEvaluationStrategy:
    """Tests for VotingEvaluationStrategy."""

    def test_initialization_valid_parameters(self):
        """Test successful initialization with valid parameters."""
        strategy = VotingEvaluationStrategy(num_votes=3)
        assert strategy.num_votes == 3

    def test_initialization_valid_tie_breaker(self):
        """Test successful initialization with valid tie_breaker."""
        strategy = VotingEvaluationStrategy(num_votes=3, tie_breaker=MAX)
        assert strategy.tie_breaker == MAX

        strategy_with_callable = VotingEvaluationStrategy(
            num_votes=3,
            tie_breaker=lambda candidates, all_scores: candidates[0],
        )
        assert callable(strategy_with_callable.tie_breaker)

    def test_initialization_invalid_num_votes(self):
        """Test initialization raises error with invalid num_votes."""
        with pytest.raises(ValueError, match="num_votes must be at least 2"):
            VotingEvaluationStrategy(num_votes=1)

    def test_initialization_invalid_tie_breaker(self):
        """Test initialization raises error with invalid tie_breaker."""
        with pytest.raises(
            ValueError,
            match="tie_breaker must be one of .* or a callable",
        ):
            VotingEvaluationStrategy(num_votes=3, tie_breaker="median")

        with pytest.raises(
            ValueError,
            match="tie_breaker must be one of .* or a callable",
        ):
            VotingEvaluationStrategy(num_votes=3, tie_breaker=123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_execute_with_most_frequent_score(self):
        """Test that VotingEvaluationStrategy returns the most frequent score."""
        strategy = VotingEvaluationStrategy(num_votes=3)

        # Mock function that returns varying results but with a majority vote for 0.8
        call_count = 0
        results = [0.8, 0.8, 0.6]  # 0.8 appears most frequently

        async def mock_call_fn():
            nonlocal call_count
            result = results[call_count % len(results)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        result = await strategy.execute(mock_call_fn)
        # Should return a GraderScore with the most common score value (0.8)
        assert result.score == 0.8
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_execute_with_grader_scores(self):
        """Test voting strategy with GraderScore objects."""
        strategy = VotingEvaluationStrategy(num_votes=3)

        # Create scores where 0.7 appears most frequently
        scores = [
            GraderScore(name="test", score=0.7, reason="First"),
            GraderScore(name="test", score=0.7, reason="Second"),
            GraderScore(name="test", score=0.5, reason="Third"),
        ]

        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return result

        result = await strategy.execute(mock_call_fn)
        # Should return a GraderScore with the most common score value (0.7)
        assert result.score == 0.7
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_execute_with_different_grader_scores(self):
        """Test voting strategy with different GraderScore objects."""
        strategy = VotingEvaluationStrategy(num_votes=5)

        # Create scores where 0.9 appears most frequently (3 out of 5 times)
        scores = [
            GraderScore(name="test", score=0.9, reason="First"),
            GraderScore(name="test", score=0.9, reason="Second"),
            GraderScore(name="test", score=0.9, reason="Third"),
            GraderScore(name="test", score=0.5, reason="Fourth"),
            GraderScore(name="test", score=0.6, reason="Fifth"),
        ]

        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return result

        result = await strategy.execute(mock_call_fn)
        # Should return a GraderScore with the most common score value (0.9)
        assert result.score == 0.9
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_execute_with_non_grader_score_raises_error(self):
        """Test that executing with non-GraderScore results raises an error."""
        strategy = VotingEvaluationStrategy(num_votes=3)

        async def mock_call_fn():
            return GraderError(name="test", reason="Test result", error="Test error")

        with pytest.raises(
            ValueError,
            match="VotingEvaluationStrategy only supports GraderScore."
            "No results were returned from the evaluation correctly.",
        ):
            await strategy.execute(mock_call_fn)

    @pytest.mark.asyncio
    async def test_execute_tie_breaker_min(self):
        """Test tie is resolved by choosing lower score with tie_breaker=min."""
        strategy = VotingEvaluationStrategy(num_votes=5, tie_breaker=MIN)
        scores = [1.0, 3.0, 5.0, 5.0, 3.0]
        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        result = await strategy.execute(mock_call_fn)
        assert result.score == 3.0

    @pytest.mark.asyncio
    async def test_execute_tie_breaker_max(self):
        """Test tie is resolved by choosing higher score with tie_breaker=max."""
        strategy = VotingEvaluationStrategy(num_votes=5, tie_breaker=MAX)
        scores = [1.0, 3.0, 5.0, 5.0, 3.0]
        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        result = await strategy.execute(mock_call_fn)
        assert result.score == 5.0

    @pytest.mark.asyncio
    async def test_execute_tie_breaker_closest_to_mean(self):
        """Test tie is resolved by choosing score closest to mean."""
        strategy = VotingEvaluationStrategy(num_votes=5, tie_breaker=CLOSEST_TO_MEAN)
        scores = [1.0, 3.0, 5.0, 5.0, 3.0]
        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        result = await strategy.execute(mock_call_fn)
        assert result.score == 3.0

    @pytest.mark.asyncio
    async def test_execute_tie_breaker_callable(self):
        """Test tie is resolved by custom callable."""

        def custom_breaker(candidates, all_scores):
            return max(candidates)

        strategy = VotingEvaluationStrategy(num_votes=5, tie_breaker=custom_breaker)
        scores = [1.0, 3.0, 5.0, 5.0, 3.0]
        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        result = await strategy.execute(mock_call_fn)
        assert result.score == 5.0

    @pytest.mark.asyncio
    async def test_execute_tie_breaker_callable_invalid_result(self):
        """Test custom tie_breaker must return one tie candidate."""

        def invalid_breaker(candidates, all_scores):
            del candidates, all_scores
            return -1.0

        strategy = VotingEvaluationStrategy(num_votes=5, tie_breaker=invalid_breaker)
        scores = [1.0, 3.0, 5.0, 5.0, 3.0]
        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count % len(scores)]
            call_count += 1
            return GraderScore(name="test", score=result, reason="Test result")

        with pytest.raises(
            ValueError,
            match="Custom tie_breaker must return one of the tie candidates.",
        ):
            await strategy.execute(mock_call_fn)
