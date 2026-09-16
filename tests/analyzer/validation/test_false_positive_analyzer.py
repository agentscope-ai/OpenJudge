# -*- coding: utf-8 -*-
"""Tests for the FalsePositiveAnalyzer."""

import pytest

from openjudge.analyzer.validation import (
    FalsePositiveAnalysisResult,
    FalsePositiveAnalyzer,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestFalsePositiveAnalyzer:
    """Test suite for FalsePositiveAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = FalsePositiveAnalyzer()
        assert analyzer.name == "False Positive Analysis"

    def test_analyze_no_false_positives(self):
        """Test analyze method with no false positives."""
        # Prepare test data - no false positives
        dataset = [
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # TN - correctly predicted negative
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # TN - correctly predicted negative
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = FalsePositiveAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, FalsePositiveAnalysisResult)
        assert result.name == "False Positive Analysis"
        assert result.false_positive_rate == 0.0  # No false positives
        assert "explanation" in result.metadata
        assert "False Positives: 0, True Negatives: 2" in result.metadata["explanation"]

    def test_analyze_some_false_positives(self):
        """Test analyze method with some false positives."""
        # Prepare test data with false positives
        dataset = [
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # FP - incorrectly predicted positive
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # TN - correctly predicted negative
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = FalsePositiveAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 1 FP out of 2 negative samples => FPR = 0.5
        assert isinstance(result, FalsePositiveAnalysisResult)
        assert result.name == "False Positive Analysis"
        assert result.false_positive_rate == 0.5
        assert "explanation" in result.metadata
        assert "False Positives: 1, True Negatives: 1" in result.metadata["explanation"]

    def test_analyze_all_false_positives(self):
        """Test analyze method where all negative samples are false positives."""
        # Prepare test data where all negative samples are incorrectly predicted as positive
        dataset = [
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # FP - incorrectly predicted positive
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # FP - incorrectly predicted positive
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),  # False positive
        ]

        # Create analyzer and run analysis
        analyzer = FalsePositiveAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 2 FP out of 2 negative samples => FPR = 1.0
        assert isinstance(result, FalsePositiveAnalysisResult)
        assert result.name == "False Positive Analysis"
        assert result.false_positive_rate == 1.0
        assert "explanation" in result.metadata
        assert "False Positives: 2, True Negatives: 0" in result.metadata["explanation"]

    def test_analyze_no_negative_samples(self):
        """Test analyze method when there are no negative samples."""
        # Prepare test data with no negative samples
        dataset = [
            {"label": 1},  # TP
            {"label": 1},  # TP
            {"label": 1},  # TP
            {"label": 1},  # TP
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=1.0, reason="Positive"),
        ]

        # Create analyzer and run analysis
        analyzer = FalsePositiveAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - No negative samples => undefined FPR, return 0
        assert isinstance(result, FalsePositiveAnalysisResult)
        assert result.name == "False Positive Analysis"
        assert result.false_positive_rate == 0.0
        assert "explanation" in result.metadata
        assert "No negative samples found for false positive rate calculation" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = FalsePositiveAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, FalsePositiveAnalysisResult)
        assert result.name == "False Positive Analysis"
        assert result.false_positive_rate == 0.0
        assert "explanation" in result.metadata
        assert (
            "No data or grader results provided for false positive rate calculation" in result.metadata["explanation"]
        )

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = FalsePositiveAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
