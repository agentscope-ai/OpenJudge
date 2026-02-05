"""Unit tests for DirectEvaluationStrategy.

This module contains tests for DirectEvaluationStrategy to ensure it functions
correctly according to the specifications.
"""

import pytest

from openjudge.evaluation_strategy.direct_evaluation_strategy import (
    DirectEvaluationStrategy,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestDirectEvaluationStrategy:
    """Tests for DirectEvaluationStrategy."""

    def test_initialization(self):
        """Test successful initialization of DirectEvaluationStrategy."""
        strategy = DirectEvaluationStrategy()
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_execute_returns_correct_result(self):
        """Test that DirectEvaluationStrategy returns the result of the call function."""
        strategy = DirectEvaluationStrategy()

        # Mock function that returns a simple value
        async def mock_call_fn(test_param="value"):
            return {"result": test_param}

        result = await strategy.execute(mock_call_fn, test_param="test_value")
        assert result == {"result": "test_value"}

    @pytest.mark.asyncio
    async def test_execute_with_grader_score(self):
        """Test that DirectEvaluationStrategy works with GraderScore objects."""
        strategy = DirectEvaluationStrategy()

        expected_score = GraderScore(name="test_grader", score=0.8, reason="Test score")

        async def mock_call_fn():
            return expected_score

        result = await strategy.execute(mock_call_fn)
        assert result == expected_score
