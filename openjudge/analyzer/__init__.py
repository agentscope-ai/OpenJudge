# -*- coding: utf-8 -*-
"""Analyzer module for computing aggregated results from evaluator outputs.

This module provides analyzers for processing evaluation results and
computing aggregated metrics, statistics, and insights.

Classes:
    AnalysisResult: Base class for analyzer results
    BaseAnalyzer: Abstract base class for analyzers
    PairwiseAnalysisResult: Result of pairwise comparison analysis
    PairwiseAnalyzer: Analyzer for pairwise comparison results
"""

from openjudge.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from openjudge.analyzer.pairwise_analyzer import (
    PairwiseAnalysisResult,
    PairwiseAnalyzer,
)

__all__ = [
    # Base classes
    "AnalysisResult",
    "BaseAnalyzer",
    # Pairwise analyzer
    "PairwiseAnalysisResult",
    "PairwiseAnalyzer",
]
