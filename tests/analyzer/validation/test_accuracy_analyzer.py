# -*- coding: utf-8 -*-
"""Tests for the AccuracyAnalyzer."""

import pytest

from openjudge.analyzer.validation import AccuracyAnalysisResult, AccuracyAnalyzer
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestAccuracyAnalyzer:
    """Test suite for AccuracyAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = AccuracyAnalyzer()
        assert analyzer.name == "Accuracy Analysis"

    def test_analyze_perfect_accuracy(self):
        """Test analyze method with perfect accuracy."""
        # Prepare test data
        dataset = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 1.0
        assert "explanation" in result.metadata
        assert "4 out of 4" in result.metadata["explanation"]

    def test_analyze_partial_accuracy(self):
        """Test analyze method with partial accuracy."""
        # Prepare test data
        dataset = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=1.0, reason="Incorrect"),  # Wrong prediction
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 0.75  # 3 out of 4 correct
        assert "explanation" in result.metadata
        assert "3 out of 4" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results" in result.metadata["explanation"]

    def test_analyze_with_none_values(self):
        """Test analyze method handling None values."""
        # Prepare test data with None values
        dataset = [
            {"label": 1},
            {"label": None},  # This sample should be skipped
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Skipped"),  # Will be skipped due to None label
            GraderScore(name="test", score=0.0, reason="Incorrect"),  # Wrong prediction
            GraderScore(name="test", score=0.0, reason="Correct"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - only 3 samples should be considered (one with None label is skipped)
        # 2 out of 3 predictions are correct (first and last ones)
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == pytest.approx(2 / 3)  # 2 out of 3 correct
        assert "explanation" in result.metadata

    def test_analyze_reports_partial_coverage(self):
        """Test analyze method excludes failed results from accuracy but reports them."""
        dataset = [
            {"label": 1},
            {"label": 0},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1, reason="Correct"),
            GraderError(name="test", reason="timeout", error="timeout"),
            GraderError(name="test", reason="rate limited", error="429"),
        ]

        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results)

        assert isinstance(result, AccuracyAnalysisResult)
        assert result.accuracy == 1.0
        assert result.metadata["total_samples"] == 3
        assert result.metadata["valid_results"] == 1
        assert result.metadata["error_results"] == 2
        assert result.metadata["coverage_rate"] == pytest.approx(1 / 3)
        assert "computed from 1 of 3 results" in result.metadata["explanation"]

    def test_analyze_full_coverage_keeps_explanation(self):
        """Test analyze method leaves the explanation untouched when every result was used."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1, reason="Correct"),
            GraderScore(name="test", score=0, reason="Correct"),
        ]

        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results)

        assert result.metadata["valid_results"] == 2
        assert result.metadata["error_results"] == 0
        assert result.metadata["coverage_rate"] == 1.0
        assert "computed from" not in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = AccuracyAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
