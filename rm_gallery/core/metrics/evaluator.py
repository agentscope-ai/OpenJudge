"""
Unified Text Similarity Evaluator

Unified text similarity evaluator providing convenient multi-metric evaluation interface.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.metrics.base import BaseMetric
from rm_gallery.core.metrics.registry import get_metric, list_available_metrics
from rm_gallery.core.metrics.schema import (
    AggregatedMetricResult,
    ComparisonInput,
    MetricResult,
)


class TextSimilarityEvaluator(BaseModel):
    """
    Unified text similarity evaluator

    Provides convenient interface for using multiple evaluation metrics.

    Attributes:
        metrics: List of metrics to use (names or configurations)
        auto_select: Whether to automatically select all available metrics
        max_workers: Maximum number of parallel worker threads for batch evaluation

    Example:
        >>> # Use specified metrics
        >>> evaluator = TextSimilarityEvaluator(
        ...     metrics=["bleu", "rouge", "fuzzy_match"]
        ... )
        >>>
        >>> # Evaluate single sample
        >>> results = evaluator.evaluate(
        ...     reference="the cat is on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>>
        >>> for metric_name, result in results.items():
        ...     print(f"{metric_name}: {result.score:.4f}")
        >>>
        >>> # Batch evaluation
        >>> batch_results = evaluator.evaluate_batch(
        ...     references=["ref1", "ref2", "ref3"],
        ...     candidates=["cand1", "cand2", "cand3"]
        ... )
    """

    metrics: List[str] = Field(default=[], description="List of metric names")
    auto_select: bool = Field(
        default=False, description="Whether to auto-select all available metrics"
    )
    max_workers: int = Field(default=8, description="Maximum number of worker threads")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize metric instances
        self._metric_instances: Dict[str, BaseMetric] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metric instances"""
        if self.auto_select:
            # Use all available metrics
            metrics_to_use = list_available_metrics()
        else:
            metrics_to_use = self.metrics

        for metric_name in metrics_to_use:
            metric_instance = get_metric(metric_name)
            if metric_instance:
                self._metric_instances[metric_name] = metric_instance
            else:
                logger.warning(f"Metric '{metric_name}' not found and will be skipped")

        logger.info(
            f"Initialized {len(self._metric_instances)} metrics: {list(self._metric_instances.keys())}"
        )

    def evaluate(
        self,
        reference: str,
        candidate: str,
        metrics: Optional[List[str]] = None,
        language: str = "en",
        normalize: bool = True,
    ) -> Dict[str, MetricResult]:
        """
        Evaluate single sample

        Args:
            reference: Reference text
            candidate: Candidate text
            metrics: Metrics to use (None means use all initialized metrics)
            language: Language code
            normalize: Whether to normalize

        Returns:
            Dict[str, MetricResult]: Mapping from metric names to results
        """
        input_data = ComparisonInput(
            reference=reference,
            candidate=candidate,
            language=language,
            normalize=normalize,
        )

        metrics_to_use = metrics or list(self._metric_instances.keys())
        results = {}

        for metric_name in metrics_to_use:
            if metric_name not in self._metric_instances:
                logger.warning(f"Metric '{metric_name}' not initialized, skipping")
                continue

            try:
                metric = self._metric_instances[metric_name]
                result = metric.compute(input_data)
                results[metric_name] = result
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
                # Return default failure result
                results[metric_name] = MetricResult(
                    name=metric_name,
                    score=0.0,
                    details={"error": str(e)},
                )

        return results

    def evaluate_batch(
        self,
        references: List[str],
        candidates: List[str],
        metrics: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        return_aggregated: bool = False,
    ) -> List[Dict[str, MetricResult]] | Dict[str, AggregatedMetricResult]:
        """
        Batch evaluation

        Args:
            references: List of reference texts
            candidates: List of candidate texts
            metrics: Metrics to use
            max_workers: Maximum number of parallel worker threads
            return_aggregated: Whether to return aggregated results

        Returns:
            List[Dict[str, MetricResult]]: List of evaluation results for each sample
            or Dict[str, AggregatedMetricResult]: Aggregated metric results
        """
        from concurrent.futures import ThreadPoolExecutor

        if len(references) != len(candidates):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs {len(candidates)} candidates"
            )

        max_workers = max_workers or self.max_workers
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.evaluate, ref, cand, metrics)
                for ref, cand in zip(references, candidates)
            ]
            results = [f.result() for f in futures]

        if return_aggregated:
            return self._aggregate_results(results)
        else:
            return results

    def _aggregate_results(
        self, results: List[Dict[str, MetricResult]]
    ) -> Dict[str, AggregatedMetricResult]:
        """
        Aggregate batch evaluation results

        Args:
            results: Batch evaluation results

        Returns:
            Dict[str, AggregatedMetricResult]: Aggregated metric results
        """
        if not results:
            return {}

        # Collect all results for each metric
        metric_results: Dict[str, List[MetricResult]] = {}
        for sample_results in results:
            for metric_name, result in sample_results.items():
                if metric_name not in metric_results:
                    metric_results[metric_name] = []
                metric_results[metric_name].append(result)

        # Create aggregated result for each metric
        aggregated = {}
        for metric_name, metric_result_list in metric_results.items():
            aggregated[metric_name] = AggregatedMetricResult.from_results(
                metric_name, metric_result_list
            )

        return aggregated

    def get_summary(
        self, results: Dict[str, MetricResult], format: str = "table"
    ) -> str:
        """
        Get evaluation results summary

        Args:
            results: Evaluation results
            format: Output format (table/json/simple)

        Returns:
            str: Formatted summary string
        """
        if format == "simple":
            lines = [f"{name}: {result.score:.4f}" for name, result in results.items()]
            return "\n".join(lines)
        elif format == "json":
            import json

            summary = {name: result.model_dump() for name, result in results.items()}
            return json.dumps(summary, indent=2)
        elif format == "table":
            # Table format
            header = f"{'Metric':<20} {'Score':<10} {'Details'}"
            separator = "-" * 60
            lines = [header, separator]

            for name, result in results.items():
                details_str = str(result.details)[:30] if result.details else ""
                line = f"{name:<20} {result.score:<10.4f} {details_str}"
                lines.append(line)

            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

    def compare_models(
        self,
        reference: str,
        candidates: Dict[str, str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, MetricResult]]:
        """
        Compare outputs from multiple models

        Args:
            reference: Reference text
            candidates: Mapping from model names to candidate texts
            metrics: Metrics to use

        Returns:
            Dict[str, Dict[str, MetricResult]]: Mapping from model names to evaluation results
        """
        comparison_results = {}

        for model_name, candidate in candidates.items():
            results = self.evaluate(reference, candidate, metrics)
            comparison_results[model_name] = results

        return comparison_results

    def get_best_model(
        self,
        comparison_results: Dict[str, Dict[str, MetricResult]],
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Select best model based on evaluation results

        Args:
            comparison_results: Return value from compare_models
            metric_weights: Metric weights (for weighted average)

        Returns:
            str: Name of the best model
        """
        if not comparison_results:
            raise ValueError("No comparison results provided")

        model_scores = {}

        for model_name, results in comparison_results.items():
            if metric_weights:
                # Weighted average
                weighted_sum = 0.0
                total_weight = 0.0
                for metric_name, result in results.items():
                    weight = metric_weights.get(metric_name, 1.0)
                    weighted_sum += result.score * weight
                    total_weight += weight
                score = weighted_sum / total_weight if total_weight > 0 else 0.0
            else:
                # Simple average
                scores = [result.score for result in results.values()]
                score = np.mean(scores) if scores else 0.0

            model_scores[model_name] = score

        # Return model with highest score
        best_model = max(model_scores, key=model_scores.get)
        logger.info(f"Best model: {best_model} (score: {model_scores[best_model]:.4f})")

        return best_model

    def list_metrics(self) -> List[str]:
        """List all initialized metrics"""
        return list(self._metric_instances.keys())

    def add_metric(self, metric_name: str, **kwargs):
        """
        Add new metric

        Args:
            metric_name: Metric name
            **kwargs: Parameters passed to metric constructor
        """
        metric = get_metric(metric_name, **kwargs)
        if metric:
            self._metric_instances[metric_name] = metric
            logger.info(f"Added metric: {metric_name}")
        else:
            logger.error(f"Failed to add metric: {metric_name}")

    def remove_metric(self, metric_name: str):
        """Remove metric"""
        if metric_name in self._metric_instances:
            del self._metric_instances[metric_name]
            logger.info(f"Removed metric: {metric_name}")
        else:
            logger.warning(f"Metric '{metric_name}' not found")
