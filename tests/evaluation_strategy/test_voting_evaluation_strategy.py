"""Unit tests for VotingEvaluationStrategy.

This module contains tests for VotingEvaluationStrategy to ensure it functions
correctly according to the specifications.
"""

import pytest

from openjudge.evaluation_strategy.voting_evaluation_strategy import (
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

    def test_initialization_invalid_num_votes(self):
        """Test initialization raises error with invalid num_votes."""
        with pytest.raises(ValueError, match="num_votes must be at least 2"):
            VotingEvaluationStrategy(num_votes=1)

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
