# -*- coding: utf-8 -*-
"""Tests for the PairwiseAnalyzer."""

import pytest

from openjudge.analyzer.pairwise_analyzer import (
    PairwiseAnalysisResult,
    PairwiseAnalyzer,
)
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.unit
class TestPairwiseAnalyzer:
    """Test suite for PairwiseAnalyzer."""

    def test_analyze_counts_only_valid_comparisons(self):
        """Test analyze method excludes failed comparisons from the total but reports them."""
        dataset = [
            {"metadata": {"model_a": "a", "model_b": "b"}},
            {"metadata": {"model_a": "a", "model_b": "b"}},
            {"metadata": {"model_a": "b", "model_b": "a"}},
        ]
        grader_results = [
            GraderScore(name="pairwise", score=1.0, reason="a wins"),
            GraderError(name="pairwise", reason="timeout", error="timeout"),
            GraderScore(name="pairwise", score=0.0, reason="a wins"),
        ]

        analyzer = PairwiseAnalyzer(model_names=["a", "b"])
        result = analyzer.analyze(dataset, grader_results)

        assert isinstance(result, PairwiseAnalysisResult)
        assert result.total_comparisons == 2
        assert result.win_rates["a"] == 1.0
        assert result.best_model == "a"
        assert result.metadata["total_samples"] == 3
        assert result.metadata["valid_results"] == 2
        assert result.metadata["error_results"] == 1
        assert "Analyzed 2 pairwise comparisons" in result.metadata["explanation"]
        assert "computed from 2 of 3 results" in result.metadata["explanation"]

    def test_analyze_length_mismatch_raises(self):
        """Test analyze method rejects a results list that does not match the dataset length."""
        dataset = [
            {"metadata": {"model_a": "a", "model_b": "b"}},
            {"metadata": {"model_a": "a", "model_b": "b"}},
        ]
        grader_results = [
            GraderScore(name="pairwise", score=1.0, reason="a wins"),
        ]

        analyzer = PairwiseAnalyzer(model_names=["a", "b"])
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(dataset, grader_results)
