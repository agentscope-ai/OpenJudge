# -*- coding: utf-8 -*-
"""Tests for the F1ScoreAnalyzer."""

import pytest

from openjudge.analyzer.validation import F1ScoreAnalysisResult, F1ScoreAnalyzer
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestF1ScoreAnalyzer:
    """Test suite for F1ScoreAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = F1ScoreAnalyzer()
        assert analyzer.name == "F1 Score Analysis"

    def test_analyze_perfect_f1_score(self):
        """Test analyze method with perfect F1 score."""
        # Prepare test data - perfect precision and recall
        dataset = [
            {"label": {1}},  # TP
            {"label": {0}},  # TN
            {"label": {1}},  # TP
            {"label": {0}},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),
        ]

        # Create analyzer and run analysis
        analyzer = F1ScoreAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, F1ScoreAnalysisResult)
        assert result.name == "F1 Score Analysis"
        assert result.f1_score == 1.0  # Perfect F1 score
        assert "explanation" in result.metadata
        assert "TP: 4, FP: 0, FN: 0" in result.metadata["explanation"]

    def test_analyze_imperfect_f1_score(self):
        """Test analyze method with imperfect F1 score."""
        # Prepare test data with some false positives and false negatives
        dataset = [
            {"label": {1}},  # TP
            {"label": {0}},  # FP
            {"label": {1}},  # FN
            {"label": {0}},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),  # False positive
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),  # False negative
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),
        ]

        # Create analyzer and run analysis
        analyzer = F1ScoreAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 1 TP, 1 FP, 1 FN => Precision=0.5, Recall=0.5, F1=0.5
        assert isinstance(result, F1ScoreAnalysisResult)
        assert result.name == "F1 Score Analysis"
        assert "explanation" in result.metadata
        assert "TP: 2, FP: 2, FN: 2" in result.metadata["explanation"]

    def test_analyze_zero_f1_score(self):
        """Test analyze method with zero F1 score."""
        # Prepare test data with no true positives
        dataset = [
            {"label": {1}},  # FN
            {"label": {0}},  # FP
            {"label": {1}},  # FN
            {"label": {0}},  # FP
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),  # False negative
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),  # False positive
            GraderScore(name="test", score=0.0, reason="Negative", metadata={"predictions": {0}}),  # False negative
            GraderScore(name="test", score=1.0, reason="Positive", metadata={"predictions": {1}}),  # False positive
        ]

        # Create analyzer and run analysis
        analyzer = F1ScoreAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 0 TP, 2 FP, 2 FN => Precision=0, Recall=0, F1=0
        assert isinstance(result, F1ScoreAnalysisResult)
        assert result.name == "F1 Score Analysis"
        assert result.f1_score == 0.0
        assert "explanation" in result.metadata
        assert "TP: 0, FP: 4, FN: 4" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = F1ScoreAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, F1ScoreAnalysisResult)
        assert result.name == "F1 Score Analysis"
        assert result.f1_score == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results provided for F1 score calculation" in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = F1ScoreAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)

    def test_analyze_counts_failed_results_as_used(self):
        """Test analyze method reports that a failed result entered the F1 calculation.

        F1 treats a result without predictions as an empty prediction set, so a
        GraderError is not excluded: it contributes false negatives and is
        counted as used, while error_results still reports it.
        """
        dataset = [
            {"label": {"a"}},
            {"label": {"b"}},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Predicted", metadata={"predictions": {"a"}}),
            GraderError(name="test", reason="timeout", error="timeout"),
        ]

        analyzer = F1ScoreAnalyzer()
        result = analyzer.analyze(dataset, grader_results)

        assert isinstance(result, F1ScoreAnalysisResult)
        assert result.metadata["false_negatives"] == 1
        assert result.metadata["total_samples"] == 2
        assert result.metadata["valid_results"] == 2
        assert result.metadata["error_results"] == 1
        assert result.metadata["skipped_results"] == 0
        assert "computed from" not in result.metadata["explanation"]
