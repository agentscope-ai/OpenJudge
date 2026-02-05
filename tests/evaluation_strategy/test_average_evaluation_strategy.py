"""Unit tests for AverageEvaluationStrategy.

This module contains tests for AverageEvaluationStrategy to ensure it functions
correctly according to the specifications.
"""

import pytest

from openjudge.evaluation_strategy.average_evaluation_strategy import (
    AverageEvaluationStrategy,
)
from openjudge.graders.schema import GraderRank, GraderScore


@pytest.mark.unit
class TestAverageEvaluationStrategy:
    """Tests for AverageEvaluationStrategy."""

    def test_initialization_valid_parameters(self):
        """Test successful initialization with valid parameters."""
        strategy = AverageEvaluationStrategy(num_evaluations=3)
        assert strategy.num_evaluations == 3

    def test_initialization_invalid_num_evaluations(self):
        """Test initialization raises error with invalid num_evaluations."""
        with pytest.raises(ValueError, match="num_evaluations must be at least 2"):
            AverageEvaluationStrategy(num_evaluations=1)

    @pytest.mark.asyncio
    async def test_execute_with_grader_scores(self):
        """Test average strategy with GraderScore objects."""
        strategy = AverageEvaluationStrategy(num_evaluations=3)

        # Create scores with different values
        scores = [
            GraderScore(name="test", score=0.5, reason="First"),
            GraderScore(name="test", score=0.7, reason="Second"),
            GraderScore(name="test", score=0.9, reason="Third"),
        ]

        call_count = 0

        async def mock_call_fn():
            nonlocal call_count
            result = scores[call_count]
            call_count += 1
            return result

        result = await strategy.execute(mock_call_fn)
        # Should return a GraderScore with the average score
        expected_average = (0.5 + 0.7 + 0.9) / 3
        assert abs(result.score - expected_average) < 0.0001  # Account for floating point precision
        assert result.name == "test"
        assert "Averaged from 3 valid evaluations out of 3 total" in result.reason

    @pytest.mark.asyncio
    async def test_execute_with_single_evaluation(self):
        """Test that AverageEvaluationStrategy works with a single evaluation."""
        # Note: This test is designed to ensure that the implementation raises an error when num_evaluations < 2
        # Since the implementation enforces num_evaluations >= 2 at initialization time,
        # we can't actually create an instance with num_evaluations=1
        # So this test verifies the constraint is properly enforced during initialization
        with pytest.raises(ValueError, match="num_evaluations must be at least 2"):
            AverageEvaluationStrategy(num_evaluations=1)

    @pytest.mark.asyncio
    async def test_execute_with_non_grader_objects_raises_error(self):
        """Test that executing with non-Grader objects raises an error."""
        strategy = AverageEvaluationStrategy(num_evaluations=3)

        async def mock_call_fn():
            return "not_a_grader_object"

        with pytest.raises(ValueError, match="No valid GraderScore results were returned from evaluation function."):
            await strategy.execute(mock_call_fn)
