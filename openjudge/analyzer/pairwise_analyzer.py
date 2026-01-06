# -*- coding: utf-8 -*-
"""Pairwise comparison analyzer for computing win rates and rankings.

This module provides the PairwiseAnalyzer class for analyzing pairwise
comparison results from LLM-based evaluations, computing win rates,
win matrices, and model rankings.

Classes:
    PairwiseAnalysisResult: Result of pairwise comparison analysis
    PairwiseAnalyzer: Analyzer for pairwise comparison results
"""

from typing import Any, Dict, List, Tuple

from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from openjudge.graders.schema import GraderResult, GraderScore


class PairwiseAnalysisResult(AnalysisResult):
    """Result of pairwise comparison analysis.

    Attributes:
        win_rates: Win rate for each model (0.0 to 1.0)
        win_matrix: Win rate matrix where win_matrix[A][B] = how often A beats B
        rankings: Model rankings sorted by win rate (descending)
        total_comparisons: Total number of pairwise comparisons
        best_model: Model with highest win rate
        worst_model: Model with lowest win rate
    """

    win_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Win rate for each model (0.0 to 1.0)",
    )
    win_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Win rate matrix: win_matrix[model_a][model_b] = how often A beats B",
    )
    rankings: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Model rankings sorted by win rate",
    )
    total_comparisons: int = Field(default=0, description="Total number of pairwise comparisons")
    best_model: str = Field(default="", description="Model with highest win rate")
    worst_model: str = Field(default="", description="Model with lowest win rate")


class PairwiseAnalyzer(BaseAnalyzer):
    """Analyzer for pairwise comparison results.

    This analyzer computes win rates and rankings from pairwise comparison results.
    It processes the results from pairwise LLM evaluations where each comparison
    yields a score indicating which response is better.

    The analyzer expects dataset samples to contain metadata with 'model_a' and
    'model_b' keys indicating which models produced the compared responses.

    Attributes:
        name: Name of the analyzer
        model_names: List of all model names being compared

    Example:
        >>> analyzer = PairwiseAnalyzer(model_names=["gpt-4", "claude", "gemini"])
        >>> result = analyzer.analyze(dataset, grader_results)
        >>> print(f"Best model: {result.best_model}")
        >>> print(f"Rankings: {result.rankings}")
    """

    name: str = "Pairwise Win Rate Analysis"

    def __init__(self, model_names: List[str]):
        """Initialize PairwiseAnalyzer.

        Args:
            model_names: List of all model names being compared
        """
        self.model_names = model_names

    def _initialize_model_matrix(self) -> Dict[str, Dict[str, int]]:
        """Initialize nested dictionary for model comparison counts."""
        return {m: {n: 0 for n in self.model_names if n != m} for m in self.model_names}

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        **kwargs: Any,
    ) -> PairwiseAnalysisResult:
        """Analyze pairwise comparison results and compute win rates.

        This method processes the grader results from pairwise comparisons,
        counting wins for each model and computing win rates and rankings.

        The score interpretation:
        - score >= 0.5: model_a wins
        - score < 0.5: model_b wins

        Args:
            dataset: List of pairwise comparison samples. Each sample should have
                a 'metadata' dict containing 'model_a' and 'model_b' keys.
            grader_results: Grader results with scores (1.0 for first wins,
                0.0 for second wins, or values in between)
            **kwargs: Additional parameters (unused)

        Returns:
            PairwiseAnalysisResult with win rates, win matrix, and rankings

        Example:
            >>> # Dataset format
            >>> dataset = [
            ...     {"metadata": {"model_a": "gpt-4", "model_b": "claude", "order": "original"}},
            ...     {"metadata": {"model_a": "claude", "model_b": "gpt-4", "order": "swapped"}},
            ... ]
            >>> # GraderScore with score=1.0 means first model (model_a) wins
            >>> results = [GraderScore(name="pairwise", score=1.0, reason="..."), ...]
            >>> analyzer = PairwiseAnalyzer(model_names=["gpt-4", "claude"])
            >>> analysis = analyzer.analyze(dataset, results)
        """
        # Pre-extract all metadata to avoid repeated dict lookups
        metadata_list = [sample.get("metadata", {}) for sample in dataset]

        # Initialize win counts (use integers for counting)
        win_counts = self._initialize_model_matrix()
        comparison_counts = self._initialize_model_matrix()

        # Use zip to pair results with metadata in one pass
        for metadata, result in zip(metadata_list, grader_results):
            model_a = metadata.get("model_a")
            model_b = metadata.get("model_b")

            if not model_a or not model_b or not isinstance(result, GraderScore):
                continue

            # score >= 0.5 means model_a wins, otherwise model_b wins
            if result.score >= 0.5:
                win_counts[model_a][model_b] += 1
            else:
                win_counts[model_b][model_a] += 1

            # Both models participated in this comparison
            comparison_counts[model_a][model_b] += 1
            comparison_counts[model_b][model_a] += 1

        # Calculate win matrix in single comprehension
        win_matrix = {
            model_a: {
                model_b: (
                    win_counts[model_a][model_b] / comparison_counts[model_a][model_b]
                    if comparison_counts[model_a][model_b] > 0
                    else 0.0
                )
                for model_b in self.model_names
                if model_a != model_b
            }
            for model_a in self.model_names
        }

        # Calculate win rates using comprehension
        win_rates = {
            model: (
                sum(win_counts[model].values()) / sum(comparison_counts[model].values())
                if sum(comparison_counts[model].values()) > 0
                else 0.0
            )
            for model in self.model_names
        }

        # Sort by win rate (descending)
        rankings = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

        return PairwiseAnalysisResult(
            name=self.name,
            win_rates=win_rates,
            win_matrix=win_matrix,
            rankings=rankings,
            total_comparisons=len(grader_results),
            best_model=rankings[0][0] if rankings else "",
            worst_model=rankings[-1][0] if rankings else "",
            metadata={
                "num_models": len(self.model_names),
                "explanation": (
                    f"Analyzed {len(grader_results)} pairwise comparisons " f"across {len(self.model_names)} models"
                ),
            },
        )
