# -*- coding: utf-8 -*-
"""
Tests for the WeightedSumAggregator class.
"""

from typing import Dict

import pytest

from openjudge.graders.schema import GraderError, GraderRank, GraderResult, GraderScore
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator


@pytest.mark.unit
class TestWeightedSumAggregator:
    """Test suite for WeightedSumAggregator"""

    def test_initialization(self):
        """Test successful initialization with and without weights"""
        # Test with weights
        weights = {"grader1": 0.6, "grader2": 0.4}
        aggregator = WeightedSumAggregator(name="test_agg", weights=weights)
        assert aggregator.name == "test_agg"
        assert aggregator.weights == weights

        # Test without weights
        aggregator_no_weights = WeightedSumAggregator(name="test_agg2")
        assert aggregator_no_weights.name == "test_agg2"
        assert aggregator_no_weights.weights == {}

    def test_empty_results(self):
        """Test aggregation with empty results"""
        aggregator = WeightedSumAggregator(name="test_agg")
        result = aggregator(grader_results={})

        assert isinstance(result, GraderError)
        assert result.name == "test_agg"
        assert result.reason == "No grader result to aggregate"
        assert result.error == "No grader result provided for aggregation"

    def test_equal_weight_aggregation(self):
        """Test aggregation with equal weights (default behavior)"""
        aggregator = WeightedSumAggregator(name="test_agg")

        results: Dict[str, GraderResult] = {
            "grader1": GraderScore(name="grader1", score=8.0, reason="Good response"),
            "grader2": GraderScore(name="grader2", score=6.0, reason="Average response"),
            "grader3": GraderScore(name="grader3", score=4.0, reason="Poor response"),
        }

        aggregated_result = aggregator(grader_results=results)

        assert isinstance(aggregated_result, GraderScore)
        assert aggregated_result.name == "test_agg"
        # Expected: (8.0 + 6.0 + 4.0) / 3 = 6.0
        assert aggregated_result.score == pytest.approx(6.0)
        assert "Weighted sum aggregation" in aggregated_result.reason

    def test_weighted_aggregation(self):
        """Test aggregation with custom weights"""
        weights = {"grader1": 0.5, "grader2": 0.3, "grader3": 0.2}
        aggregator = WeightedSumAggregator(name="test_agg", weights=weights)

        results: Dict[str, GraderResult] = {
            "grader1": GraderScore(name="grader1", score=10.0, reason="Excellent"),
            "grader2": GraderScore(name="grader2", score=5.0, reason="Average"),
            "grader3": GraderScore(name="grader3", score=0.0, reason="Poor"),
        }

        aggregated_result = aggregator(grader_results=results)

        assert isinstance(aggregated_result, GraderScore)
        assert aggregated_result.name == "test_agg"
        # Expected: (10.0 * 0.5 + 5.0 * 0.3 + 0.0 * 0.2) / (0.5 + 0.3 + 0.2) = 6.5
        assert aggregated_result.score == pytest.approx(6.5)
        assert "Weighted sum aggregation" in aggregated_result.reason

    def test_mixed_result_types(self):
        """Test aggregation with mixed result types (scores, errors, ranks)"""
        aggregator = WeightedSumAggregator(name="test_agg")

        results: Dict[str, GraderResult] = {
            "score_grader": GraderScore(name="score_grader", score=8.0, reason="Good"),
            "error_grader": GraderError(name="error_grader", error="Network error", reason="Failed"),
            "rank_grader": GraderRank(name="rank_grader", rank=[1, 2, 3], reason="Ranked"),
        }

        aggregated_result = aggregator(grader_results=results)

        assert isinstance(aggregated_result, GraderScore)
        assert aggregated_result.name == "test_agg"
        # Only the GraderScore should contribute to the final score
        assert aggregated_result.score == pytest.approx(8.0)
        assert "score_grader: 8.000" in aggregated_result.reason
        assert "error_grader: ERROR" in aggregated_result.reason
        assert "rank_grader: rank [1, 2, 3]" in aggregated_result.reason

    def test_zero_weight_aggregation(self):
        """Test aggregation when all weights are zero"""
        weights = {"grader1": 0.0, "grader2": 0.0}
        aggregator = WeightedSumAggregator(name="test_agg", weights=weights)

        results: Dict[str, GraderResult] = {
            "grader1": GraderScore(name="grader1", score=10.0, reason="Excellent"),
            "grader2": GraderScore(name="grader2", score=5.0, reason="Average"),
        }

        aggregated_result = aggregator(grader_results=results)

        assert isinstance(aggregated_result, GraderScore)
        assert aggregated_result.name == "test_agg"
        # With zero weights, final score should be 0.0
        assert aggregated_result.score == pytest.approx(0.0)

    def test_missing_weights(self):
        """Test aggregation when some graders don't have specified weights"""
        weights = {"grader1": 0.6}  # Only weight for grader1 specified
        aggregator = WeightedSumAggregator(name="test_agg", weights=weights)

        results: Dict[str, GraderResult] = {
            "grader1": GraderScore(name="grader1", score=10.0, reason="Excellent"),
            "grader2": GraderScore(name="grader2", score=5.0, reason="Average"),
        }

        # grader2 should get default weight of 0.0
        aggregated_result = aggregator(grader_results=results)

        assert isinstance(aggregated_result, GraderScore)
        assert aggregated_result.name == "test_agg"
        # Only grader1 contributes: (10.0 * 0.6) / 0.6 = 10.0
        assert aggregated_result.score == pytest.approx(10.0)
