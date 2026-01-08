# -*- coding: utf-8 -*-
"""
Implementation of weighted sum aggregator for combining grader results.
"""

from typing import Dict

from openjudge.graders.schema import GraderError, GraderRank, GraderResult, GraderScore
from openjudge.runner.aggregator.base_aggregator import BaseAggregator


class WeightedSumAggregator(BaseAggregator):
    """
    Aggregator that combines grader scores using weighted sum.

    This aggregator takes numerical scores from multiple graders and combines
    them using provided weights to produce a single aggregated score for one sample.
    """

    def __init__(self, name: str, weights: Dict[str, float] = None):
        """
        Initialize the weighted sum aggregator.

        Args:
            name: Name of the aggregator
            weights: Dictionary mapping grader names to weights. If None, equal weights are used.
        """
        super().__init__(name)
        self.weights = weights or {}

    def __call__(self, grader_results: Dict[str, GraderResult], **kwargs) -> GraderResult:
        """
        Aggregate multiple grader results using weighted sum for a single sample.

        Args:
            grader_results: Dictionary mapping grader names to GraderResult objects for a single sample
            **kwargs: Additional arguments (unused)

        Returns:
            Aggregated result as a GraderResult object
        """
        if not grader_results:
            return GraderError(
                name=self.name,
                reason="No grader result to aggregate",
                error="No grader result provided for aggregation",
            )

        # Initialize weights if not provided (equal weights)
        if not self.weights:
            grader_names = list(grader_results.keys())
            equal_weight = 1.0 / len(grader_names) if grader_names else 0.0
            weights = {name: equal_weight for name in grader_names}
        else:
            weights = self.weights

        weighted_sum = 0.0
        total_weight = 0.0
        component_scores = {}

        # Collect scores from all graders for this sample
        for grader_name, result in grader_results.items():
            # Only process results of GraderScore type (skip errors, ranks, etc.)
            if isinstance(result, GraderScore):
                weight = weights.get(grader_name, 0.0)
                weighted_sum += result.score * weight
                total_weight += weight
                component_scores[grader_name] = {
                    "score": result.score,
                    "weight": weight,
                    "reason": result.reason,
                }
            elif isinstance(result, GraderError):
                component_scores[grader_name] = {
                    "error": result.error,
                }
            elif isinstance(result, GraderRank):
                component_scores[grader_name] = {
                    "rank": result.rank,  # Fix: use 'rank' instead of 'ranking'
                    "reason": result.reason,
                }

        # Calculate final aggregated score
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.0

        # Create a descriptive reason for the aggregation
        reason_parts = []
        for name, info in component_scores.items():
            if "score" in info:
                reason_parts.append(
                    f"{name}: {info['score']:.3f} (weight: {info['weight']:.2f})",
                )
            elif "error" in info:
                reason_parts.append(f"{name}: ERROR")
            elif "rank" in info:
                reason_parts.append(f"{name}: rank {info['rank']}")

        reason = f"Weighted sum aggregation of {', '.join(reason_parts)}"

        # Return the aggregated result as a GraderScore
        return GraderScore(
            name=self.name,
            score=final_score,
            reason=reason,
            metadata={
                "weights": weights,
            },
        )
