# -*- coding: utf-8 -*-
"""Tests for the PrecisionAnalyzer."""

import pytest

from openjudge.analyzer.validation import PrecisionAnalysisResult, PrecisionAnalyzer
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestPrecisionAnalyzer:
    """Test suite for PrecisionAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = PrecisionAnalyzer()
        assert analyzer.name == "Precision Analysis"

    def test_analyze_perfect_precision(self):
        """Test analyze method with perfect precision."""
        # Prepare test data - all positive predictions are correct
        dataset = [
            {"label": 1},  # TP
            {"label": 0},  # TN
            {"label": 1},  # TP
            {"label": 0},  # FP
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
        ]

        # Create analyzer and run analysis
        analyzer = PrecisionAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, PrecisionAnalysisResult)
        assert result.name == "Precision Analysis"
        assert "explanation" in result.metadata
        assert "True Positives: 2, False Positives: 1" in result.metadata["explanation"]

    def test_analyze_imperfect_precision(self):
        """Test analyze method with imperfect precision."""
        # Prepare test data with some false positives
        dataset = [
            {"label": 1},  # TP
            {"label": 0},  # FP
            {"label": 1},  # TP
            {"label": 0},  # FP
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
        ]

        # Create analyzer and run analysis
        analyzer = PrecisionAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 2 TP, 2 FP => Precision = 2/(2+2) = 0.5
        assert isinstance(result, PrecisionAnalysisResult)
        assert result.name == "Precision Analysis"
        assert result.precision == 0.5
        assert "explanation" in result.metadata
        assert "True Positives: 2, False Positives: 2" in result.metadata["explanation"]

    def test_analyze_no_positive_predictions(self):
        """Test analyze method when there are no positive predictions."""
        # Prepare test data - no positive predictions
        dataset = [
            {"label": 1},  # FN - missed positive
            {"label": 0},  # TN
            {"label": 1},  # FN - missed positive
            {"label": 0},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = PrecisionAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - No positive predictions => undefined precision, return 0
        assert isinstance(result, PrecisionAnalysisResult)
        assert result.name == "Precision Analysis"
        assert result.precision == 0.0
        assert "explanation" in result.metadata
        assert "No positive predictions found for precision calculation" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = PrecisionAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, PrecisionAnalysisResult)
        assert result.name == "Precision Analysis"
        assert result.precision == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results provided for precision calculation" in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = PrecisionAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)

    def test_analyze_reports_partial_coverage(self):
        """Test analyze method excludes failed results from precision but reports them."""
        dataset = [
            {"label": 1},
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderError(name="test", reason="timeout", error="timeout"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        analyzer = PrecisionAnalyzer()
        result = analyzer.analyze(dataset, grader_results)

        assert isinstance(result, PrecisionAnalysisResult)
        assert result.precision == 1.0
        assert result.metadata["total_samples"] == 3
        assert result.metadata["valid_results"] == 2
        assert result.metadata["error_results"] == 1
        assert "computed from 2 of 3 results (1 errors, 0 skipped)" in result.metadata["explanation"]
