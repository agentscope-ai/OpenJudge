# -*- coding: utf-8 -*-
"""Tests for the FalseNegativeAnalyzer."""

import pytest

from openjudge.analyzer.validation import (
    FalseNegativeAnalysisResult,
    FalseNegativeAnalyzer,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestFalseNegativeAnalyzer:
    """Test suite for FalseNegativeAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = FalseNegativeAnalyzer()
        assert analyzer.name == "False Negative Analysis"

    def test_analyze_no_false_negatives(self):
        """Test analyze method with no false negatives."""
        # Prepare test data - no false negatives
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
        analyzer = FalseNegativeAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, FalseNegativeAnalysisResult)
        assert result.name == "False Negative Analysis"
        assert result.false_negative_rate == 0.0  # No false negatives
        assert "explanation" in result.metadata
        assert "False Negatives: 0, True Positives: 2" in result.metadata["explanation"]

    def test_analyze_some_false_negatives(self):
        """Test analyze method with some false negatives."""
        # Prepare test data with false negatives
        dataset = [
            {"label": 1},  # FN - incorrectly predicted negative
            {"label": 0},  # TN - correctly predicted negative
            {"label": 1},  # TP - correctly predicted positive
            {"label": 0},  # TN - correctly predicted negative
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative"),  # False negative
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = FalseNegativeAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 1 FN out of 2 positive samples => FNR = 0.5
        assert isinstance(result, FalseNegativeAnalysisResult)
        assert result.name == "False Negative Analysis"
        assert result.false_negative_rate == 0.5
        assert "explanation" in result.metadata
        assert "False Negatives: 1, True Positives: 1" in result.metadata["explanation"]

    def test_analyze_all_false_negatives(self):
        """Test analyze method where all positive samples are false negatives."""
        # Prepare test data where all positive samples are incorrectly predicted as negative
        dataset = [
            {"label": 1},  # FN - incorrectly predicted negative
            {"label": 0},  # TN - correctly predicted negative
            {"label": 1},  # FN - incorrectly predicted negative
            {"label": 0},  # TN - correctly predicted negative
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative"),  # False negative
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),  # False negative
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = FalseNegativeAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 2 FN out of 2 positive samples => FNR = 1.0
        assert isinstance(result, FalseNegativeAnalysisResult)
        assert result.name == "False Negative Analysis"
        assert result.false_negative_rate == 1.0
        assert "explanation" in result.metadata
        assert "False Negatives: 2, True Positives: 0" in result.metadata["explanation"]

    def test_analyze_no_positive_samples(self):
        """Test analyze method when there are no positive samples."""
        # Prepare test data with no positive samples
        dataset = [
            {"label": 0},  # TN
            {"label": 0},  # TN
            {"label": 0},  # TN
            {"label": 0},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = FalseNegativeAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - No positive samples => undefined FNR, return 0
        assert isinstance(result, FalseNegativeAnalysisResult)
        assert result.name == "False Negative Analysis"
        assert result.false_negative_rate == 0.0
        assert "explanation" in result.metadata
        assert "No positive samples found for false negative rate calculation" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = FalseNegativeAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, FalseNegativeAnalysisResult)
        assert result.name == "False Negative Analysis"
        assert result.false_negative_rate == 0.0
        assert "explanation" in result.metadata
        assert (
            "No data or grader results provided for false negative rate calculation" in result.metadata["explanation"]
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

        analyzer = FalseNegativeAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
