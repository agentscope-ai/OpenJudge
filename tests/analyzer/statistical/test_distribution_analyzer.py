# -*- coding: utf-8 -*-
"""Tests for the DistributionAnalyzer."""


import pytest

from openjudge.analyzer.statistical.distribution_analyzer import (
    DistributionAnalysisResult,
    DistributionAnalyzer,
)
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestDistributionAnalyzer:
    """Test suite for DistributionAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = DistributionAnalyzer()
        assert analyzer.name == "Distribution Analysis"

    def test_analyze_empty_results(self):
        """Test analyzing with empty results."""
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze([], [])

        assert isinstance(result, DistributionAnalysisResult)
        assert result.name == "Distribution Analysis"
        assert result.mean == 0.0
        assert result.median == 0.0
        assert result.stdev == 0.0
        assert result.min_score == 0.0
        assert result.max_score == 0.0
        assert "explanation" in result.metadata
        assert result.metadata["explanation"] == "No grader results provided for distribution calculation"

    def test_analyze_no_valid_scores(self):
        """Test analyzing with invalid scores."""
        analyzer = DistributionAnalyzer()

        # Create objects that don't have a score attribute
        class MockResult:
            pass

        grader_results = [
            MockResult(),
            MockResult(),
        ]
        result = analyzer.analyze([], grader_results)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.name == "Distribution Analysis"
        assert result.mean == 0.0
        assert result.median == 0.0
        assert result.stdev == 0.0
        assert result.min_score == 0.0
        assert result.max_score == 0.0
        assert "explanation" in result.metadata
        assert result.metadata["explanation"] == "No valid scores found in grader results"

    def test_analyze_single_score(self):
        """Test analyzing with a single score."""
        analyzer = DistributionAnalyzer()
        grader_results = [
            GraderScore(name="test", score=5.0, reason="Good"),
        ]
        result = analyzer.analyze([{}], grader_results)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.name == "Distribution Analysis"
        assert result.mean == 5.0
        assert result.median == 5.0
        assert result.stdev == 0.0  # Standard deviation is 0 for single value
        assert result.min_score == 5.0
        assert result.max_score == 5.0
        assert "explanation" in result.metadata
        assert "total_samples" in result.metadata
        assert result.metadata["total_samples"] == 1

    def test_analyze_multiple_scores(self):
        """Test analyzing with multiple scores."""
        analyzer = DistributionAnalyzer()
        grader_results = [
            GraderScore(name="test1", score=1.0, reason="Poor"),
            GraderScore(name="test2", score=5.0, reason="Good"),
            GraderScore(name="test3", score=3.0, reason="Average"),
            GraderScore(name="test4", score=7.0, reason="Excellent"),
            GraderScore(name="test5", score=9.0, reason="Outstanding"),
        ]
        result = analyzer.analyze([{}] * 5, grader_results)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.name == "Distribution Analysis"
        assert result.mean == 5.0  # (1+5+3+7+9)/5 = 5
        assert result.median == 5.0  # Middle value of sorted [1,3,5,7,9]
        assert result.stdev > 0  # Should be positive with multiple different values
        assert result.min_score == 1.0
        assert result.max_score == 9.0
        assert "explanation" in result.metadata
        assert "total_samples" in result.metadata
        assert result.metadata["total_samples"] == 5

    def test_analyze_float_scores(self):
        """Test analyzing with float scores."""
        analyzer = DistributionAnalyzer()
        grader_results = [
            GraderScore(name="test1", score=1.5, reason="Below Average"),
            GraderScore(name="test2", score=2.7, reason="Average"),
            GraderScore(name="test3", score=3.9, reason="Above Average"),
        ]
        result = analyzer.analyze([{}] * 3, grader_results)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.name == "Distribution Analysis"
        assert result.mean == pytest.approx(2.7)  # (1.5+2.7+3.9)/3 = 2.7
        assert result.median == 2.7  # Middle value of sorted [1.5,2.7,3.9]
        assert result.min_score == 1.5
        assert result.max_score == 3.9
        assert "explanation" in result.metadata
        assert "total_samples" in result.metadata
        assert result.metadata["total_samples"] == 3

    def test_analyze_reports_partial_coverage(self):
        """Test analyze method excludes failed results from the statistics but reports them."""
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Good"),
            GraderError(name="test", reason="timeout", error="timeout"),
            GraderError(name="test", reason="rate limited", error="429"),
        ]

        analyzer = DistributionAnalyzer()
        result = analyzer.analyze([], grader_results)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.mean == 1.0
        assert result.metadata["total_samples"] == 3
        assert result.metadata["valid_results"] == 1
        assert result.metadata["error_results"] == 2
        assert result.metadata["skipped_results"] == 0
        assert result.metadata["coverage_rate"] == pytest.approx(1 / 3)
        assert "computed from 1 of 3 results (2 errors, 0 skipped)" in result.metadata["explanation"]
