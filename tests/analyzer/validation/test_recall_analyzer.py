# -*- coding: utf-8 -*-
"""Tests for the RecallAnalyzer."""

import pytest

from openjudge.analyzer.validation import RecallAnalysisResult, RecallAnalyzer
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestRecallAnalyzer:
    """Test suite for RecallAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = RecallAnalyzer()
        assert analyzer.name == "Recall Analysis"

    def test_analyze_perfect_recall(self):
        """Test analyze method with perfect recall."""
        # Prepare test data - all actual positives are correctly identified
        dataset = [
            {"label": 1},  # TP
            {"label": 0},  # TN
            {"label": 1},  # TP
            {"label": 0},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = RecallAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, RecallAnalysisResult)
        assert result.name == "Recall Analysis"
        assert result.recall == 1.0  # Perfect recall
        assert "explanation" in result.metadata
        assert "True Positives: 2, False Negatives: 0" in result.metadata["explanation"]

    def test_analyze_imperfect_recall(self):
        """Test analyze method with imperfect recall."""
        # Prepare test data with some false negatives
        dataset = [
            {"label": 1},  # FN - missed positive
            {"label": 0},  # TN
            {"label": 1},  # TP
            {"label": 0},  # TN
        ]

        grader_results = [
            GraderScore(name="test", score=0.0, reason="Negative"),  # False negative
            GraderScore(name="test", score=0.0, reason="Negative"),
            GraderScore(name="test", score=1.0, reason="Positive"),
            GraderScore(name="test", score=0.0, reason="Negative"),
        ]

        # Create analyzer and run analysis
        analyzer = RecallAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - 1 TP, 1 FN => Recall = 1/(1+1) = 0.5
        assert isinstance(result, RecallAnalysisResult)
        assert result.name == "Recall Analysis"
        assert result.recall == 0.5
        assert "explanation" in result.metadata
        assert "True Positives: 1, False Negatives: 1" in result.metadata["explanation"]

    def test_analyze_no_actual_positives(self):
        """Test analyze method when there are no actual positives."""
        # Prepare test data with no actual positives
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
        analyzer = RecallAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - No actual positives => undefined recall, return 0
        assert isinstance(result, RecallAnalysisResult)
        assert result.name == "Recall Analysis"
        assert result.recall == 0.0
        assert "explanation" in result.metadata
        assert "No actual positives found for recall calculation" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = RecallAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, RecallAnalysisResult)
        assert result.name == "Recall Analysis"
        assert result.recall == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results provided for recall calculation" in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"label": 1},
            {"label": 0},
        ]
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
        ]

        analyzer = RecallAnalyzer()
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
