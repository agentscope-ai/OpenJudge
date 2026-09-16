# -*- coding: utf-8 -*-
"""Tests for the CorrelationAnalyzer."""

import pytest

from openjudge.analyzer.validation import CorrelationAnalysisResult, CorrelationAnalyzer
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestCorrelationAnalyzer:
    """Test suite for CorrelationAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = CorrelationAnalyzer()
        assert analyzer.name == "Correlation Analysis"

    def test_analyze_perfect_positive_correlation(self):
        """Test analyze method with perfect positive correlation."""
        # Prepare test data with perfect positive correlation
        dataset = [
            {"label": 1.0},
            {"label": 2.0},
            {"label": 3.0},
            {"label": 4.0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Score 1"),
            GraderScore(name="test", score=2.0, reason="Score 2"),
            GraderScore(name="test", score=3.0, reason="Score 3"),
            GraderScore(name="test", score=4.0, reason="Score 4"),
        ]

        # Create analyzer and run analysis
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, CorrelationAnalysisResult)
        assert result.name == "Correlation Analysis"
        assert result.correlation == 1.0  # Perfect positive correlation
        assert "explanation" in result.metadata
        assert "Correlation based on 4 data points: 1.0000" in result.metadata["explanation"]

    def test_analyze_perfect_negative_correlation(self):
        """Test analyze method with perfect negative correlation."""
        # Prepare test data with perfect negative correlation
        dataset = [
            {"label": 1.0},
            {"label": 2.0},
            {"label": 3.0},
            {"label": 4.0},
        ]

        grader_results = [
            GraderScore(name="test", score=4.0, reason="Score 4"),
            GraderScore(name="test", score=3.0, reason="Score 3"),
            GraderScore(name="test", score=2.0, reason="Score 2"),
            GraderScore(name="test", score=1.0, reason="Score 1"),
        ]

        # Create analyzer and run analysis
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, CorrelationAnalysisResult)
        assert result.name == "Correlation Analysis"
        assert result.correlation == -1.0  # Perfect negative correlation
        assert "explanation" in result.metadata
        assert "Correlation based on 4 data points: -1.0000" in result.metadata["explanation"]

    def test_analyze_moderate_correlation(self):
        """Test analyze method with moderate correlation."""
        # Prepare test data with moderate correlation
        dataset = [
            {"label": 1.0},
            {"label": 2.0},
            {"label": 3.0},
            {"label": 4.0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.2, reason="Score 1.2"),
            GraderScore(name="test", score=1.8, reason="Score 1.8"),
            GraderScore(name="test", score=3.2, reason="Score 3.2"),
            GraderScore(name="test", score=3.8, reason="Score 3.8"),
        ]

        # Create analyzer and run analysis
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, CorrelationAnalysisResult)
        assert result.name == "Correlation Analysis"
        assert result.correlation > 0.9  # Strong positive correlation
        assert "explanation" in result.metadata
        assert "Correlation based on 4 data points:" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, CorrelationAnalysisResult)
        assert result.name == "Correlation Analysis"
        assert result.correlation == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results provided for correlation calculation" in result.metadata["explanation"]

    def test_analyze_zero_variance_scores(self):
        """Test analyze method when predicted scores are all identical (zero variance)."""
        dataset = [
            {"label": 0.8},
            {"label": 0.6},
            {"label": 0.9},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Score 1"),
            GraderScore(name="test", score=1.0, reason="Score 2"),
            GraderScore(name="test", score=1.0, reason="Score 3"),
        ]

        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        assert isinstance(result, CorrelationAnalysisResult)
        assert result.correlation == 0.0
        assert result.correlation == result.correlation  # not NaN
        assert "explanation" in result.metadata
        assert "Correlation based on 3 data points: 0.0000" in result.metadata["explanation"]

    def test_analyze_zero_variance_labels(self):
        """Test analyze method when ground truth labels are all identical (zero variance)."""
        dataset = [
            {"label": 1.0},
            {"label": 1.0},
            {"label": 1.0},
        ]

        grader_results = [
            GraderScore(name="test", score=0.1, reason="Score 1"),
            GraderScore(name="test", score=0.5, reason="Score 2"),
            GraderScore(name="test", score=0.9, reason="Score 3"),
        ]

        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        assert isinstance(result, CorrelationAnalysisResult)
        assert result.correlation == 0.0
        assert result.correlation == result.correlation  # not NaN
        assert "explanation" in result.metadata
        assert "Correlation based on 3 data points: 0.0000" in result.metadata["explanation"]

    def test_analyze_zero_variance_scores_and_labels_matching(self):
        """Test analyze method when both scores and labels are identical and equal to each other."""
        dataset = [
            {"label": 1.0},
            {"label": 1.0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Score 1"),
            GraderScore(name="test", score=1.0, reason="Score 2"),
        ]

        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        assert isinstance(result, CorrelationAnalysisResult)
        assert result.correlation == 1.0
        assert "explanation" in result.metadata
        assert "Correlation based on 2 data points: 1.0000" in result.metadata["explanation"]

    def test_analyze_insufficient_data(self):
        """Test analyze method with insufficient data."""
        # Prepare test data with only one sample (insufficient for correlation)
        dataset = [
            {"label": 1.0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Score 1"),
        ]

        # Create analyzer and run analysis
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, CorrelationAnalysisResult)
        assert result.name == "Correlation Analysis"
        assert result.correlation == 0.0  # Insufficient data
        assert "explanation" in result.metadata
        assert "Insufficient data points for correlation calculation" in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
