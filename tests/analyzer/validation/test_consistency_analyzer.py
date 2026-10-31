# -*- coding: utf-8 -*-
"""Tests for the ConsistencyAnalyzer."""

import pytest

from openjudge.analyzer.statistical import (
    ConsistencyAnalysisResult,
    ConsistencyAnalyzer,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestConsistencyAnalyzer:
    """Test suite for ConsistencyAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = ConsistencyAnalyzer()
        assert analyzer.name == "Consistency Analysis"

    def test_analyze_perfect_consistency(self):
        """Test analyze method with perfect consistency."""
        # Prepare test data - two identical sets of results
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == 1.0
        assert "explanation" in result.metadata
        assert "4 paired evaluations" in result.metadata["explanation"]

    def test_analyze_partial_consistency(self):
        """Test analyze method with partial consistency."""
        # Prepare test data with some differences
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=1.0, reason="Incorrect"),  # Different from first run
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert "explanation" in result.metadata
        assert "4 paired evaluations" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], [], [])

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == 0.0
        assert "explanation" in result.metadata
        assert "No results provided for consistency calculation" in result.metadata["explanation"]

    def test_analyze_mismatched_lengths(self):
        """Test analyze method with mismatched lengths."""
        # Prepare test data with different lengths
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Extra"),  # Extra result
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions - should only compare first 2 results
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == pytest.approx(1.0)  # First 2 results are consistent
        assert "explanation" in result.metadata
        assert "2 paired evaluations" in result.metadata["explanation"]

    @pytest.mark.parametrize("second_scores, expected", [([0.0, 1.0, 2.0], 1.0), ([2.0, 1.0, 0.0], -1.0)])
    def test_legacy_two_argument_call(self, second_scores, expected):
        """The documented two-result-list call agrees with the current API."""
        first = [GraderScore(name="test", score=score, reason="ok") for score in [0.0, 1.0, 2.0]]
        second = [GraderScore(name="test", score=score, reason="ok") for score in second_scores]
        analyzer = ConsistencyAnalyzer()

        legacy = analyzer.analyze(first, second)
        current = analyzer.analyze([{}, {}, {}], first, second)

        assert legacy.consistency == pytest.approx(expected)
        assert legacy == current
        assert legacy.metadata["paired_evaluations"] == 3

    @pytest.mark.parametrize("first_empty, second_empty", [(True, False), (False, True), (True, True)])
    def test_legacy_empty_run(self, first_empty, second_empty):
        """An empty run in the legacy API yields the existing no-results report."""
        scores = [GraderScore(name="test", score=value, reason="ok") for value in [0.0, 1.0]]

        result = ConsistencyAnalyzer().analyze([] if first_empty else scores, [] if second_empty else scores)

        assert result.consistency == 0.0
        assert result.metadata["explanation"] == "No results provided for consistency calculation"

    def test_explicit_empty_second_run_is_not_a_legacy_call(self):
        """The dataset must never stand in for an explicitly empty second run."""
        first = [GraderScore(name="test", score=value, reason="ok") for value in [0.0, 1.0]]

        result = ConsistencyAnalyzer().analyze([{}, {}], first, [])

        assert result.consistency == 0.0
        assert result.metadata["explanation"] == "No results provided for consistency calculation"
