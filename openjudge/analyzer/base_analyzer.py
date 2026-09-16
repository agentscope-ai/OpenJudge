# -*- coding: utf-8 -*-
"""Base classes for analyzers that compute aggregated results from evaluator outputs.

This module defines the base classes for analyzers, which are responsible for
computing aggregated results from individual evaluator outputs. Analyzers take
the detailed results from a single evaluator and compute summary statistics.

Classes:
    AnalysisResult: Base class for analyzer results.
    BaseAnalyzer: Abstract base class for analyzers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.graders.schema import GraderError, GraderResult


class AnalysisResult(BaseModel):
    """Base class for storing analyzer computation results.

    This class serves as the foundation for all analyzer results. It contains essential
    metadata about the computation process and can be extended to include specific
    result fields for different types of analyzers.

    Attributes:
        name (str): The unique identifier name of the analyzer.
        metadata (Dict[str, Any]): Additional information about the analyzer result,
            including computation details, timestamps, input parameters, and other
            relevant contextual data.

    Examples:
        Create a basic analysis result:
        >>> result = AnalysisResult(name="test_analysis")
        >>> print(result.name)
        test_analysis

        Create an analysis with metadata:
        >>> result = AnalysisResult(
        ...     name="accuracy_analyzer",
        ...     metadata={"computed_at": "2023-01-01", "version": "1.0"}
        ... )
        >>> print(result.metadata)
        {'computed_at': '2023-01-01', 'version': '1.0'}
    """

    name: str = Field(default=..., description="The name of the analyzer")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the analyzer result",
    )


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers that compute aggregated evaluation results.

    Analyzers process detailed evaluation results from individual evaluators and
    compute aggregated metrics, statistics, and summary insights. They are
    primarily used to transform granular evaluation data into meaningful,
    high-level reports and analytics.

    As an abstract base class, this defines the standard interface that all
    concrete analyzer implementations must follow. Subclasses are required to
    implement the `analyze` method to provide specific computational logic.

    Attributes:
        name (str): The human-readable name of the analyzer, used for identification
            and reporting purposes.

    Examples:
        Creating a custom analyzer implementation:
        >>> from openjudge.graders.schema import GraderResult
        >>> class AccuracyAnalyzer(BaseAnalyzer):
        ...     name = "Accuracy Analyzer"
        ...     def analyze(self, data, grader_results, **kwargs):
        ...         # Calculate accuracy metrics from results
        ...         return AnalysisResult(
        ...             name=self.name,
        ...             metadata={
        ...                 "sample_count": len(data),
        ...                 "analysis_type": "accuracy"
        ...             }
        ...         )
        >>> analyzer = AccuracyAnalyzer()
        >>> print(analyzer.name)
        Accuracy Analyzer
    """

    name: str = "Base Analysis"

    def _validate_alignment(self, dataset: List[dict], grader_results: List[GraderResult]) -> None:
        """Raise if dataset and grader results cannot be paired positionally.

        Analyzers pair samples with results by position, so a length mismatch
        would silently drop the tail of the longer list.

        Args:
            dataset: The data samples that were evaluated.
            grader_results: The grader results, one per sample.

        Raises:
            ValueError: If the two lists have different lengths.

        Example:
            >>> analyzer = AccuracyAnalyzer()
            >>> analyzer._validate_alignment([{"label": 1}], [])
            Traceback (most recent call last):
                ...
            ValueError: dataset and grader_results must have the same length, got 1 and 0
        """
        if len(dataset) != len(grader_results):
            raise ValueError(
                f"dataset and grader_results must have the same length, got {len(dataset)} and {len(grader_results)}",
            )

    def _coverage_metadata(
        self,
        grader_results: List[GraderResult],
        valid_results: int,
        explanation: str,
    ) -> Dict[str, Any]:
        """Build metadata that states how many results actually entered the metric.

        Failed (``GraderError``) or otherwise unusable results are excluded from
        the metric itself, but the exclusion must be visible: the counts are
        returned as metadata and the explanation is extended when coverage is
        incomplete.

        Args:
            grader_results: All results that were submitted to the analyzer.
            valid_results: How many of them the analyzer actually used.
            explanation: The analyzer's explanation of the computed metric.

        Returns:
            Dict[str, Any]: ``explanation`` plus ``total_samples``, ``valid_results``,
            ``error_results``, ``skipped_results`` and ``coverage_rate``.

        Example:
            >>> from openjudge.graders.schema import GraderError, GraderScore
            >>> results = [
            ...     GraderScore(name="judge", score=1.0, reason="ok"),
            ...     GraderError(name="judge", reason="timeout", error="timeout"),
            ... ]
            >>> analyzer = DistributionAnalyzer()
            >>> metadata = analyzer._coverage_metadata(results, valid_results=1, explanation="Mean score: 1.0")
            >>> print(metadata["explanation"])
            Mean score: 1.0; computed from 1 of 2 results (1 errors, 0 skipped)
            >>> print(metadata["coverage_rate"])
            0.5
        """
        total = len(grader_results)
        errors = sum(isinstance(result, GraderError) for result in grader_results)
        skipped = max(0, total - valid_results - errors)
        if valid_results < total:
            explanation = (
                f"{explanation}; computed from {valid_results} of {total} results "
                f"({errors} errors, {skipped} skipped)"
            )
            logger.warning(f"{self.name}: only {valid_results} of {total} grader results were usable")
        return {
            "explanation": explanation,
            "total_samples": total,
            "valid_results": valid_results,
            "error_results": errors,
            "skipped_results": skipped,
            "coverage_rate": valid_results / total if total else 0.0,
        }

    @abstractmethod
    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        **kwargs: Any,
    ) -> AnalysisResult:
        """Execute the analysis computation on provided evaluation data.

        This abstract method must be implemented by all subclasses to define
        the specific logic for analyzing evaluation results. The method processes
        the input data and grader results to produce aggregated metrics and insights.

        Args:
            dataset (List[dict]): The collection of data samples that were evaluated.
                Each dictionary represents a single sample containing its input
                parameters and expected outputs.
            grader_results (List[GraderResult]): The detailed evaluation results
                from a single evaluator, structured as a list of GraderResult
                objects (one per sample).
            **kwargs: Optional configuration parameters for the analysis process.
                These may include filtering criteria, threshold values, or other
                algorithm-specific settings.

        Returns:
            AnalysisResult: An instance containing the computed analysis results,
            including aggregated metrics and relevant metadata about the computation.

        Examples:
            Implementing a concrete analyzer:
            >>> from openjudge.graders.schema import GraderResult
            >>> class PerformanceAnalyzer(BaseAnalyzer):
            ...     name = "Performance Analyzer"
            ...     def analyze(self, data, grader_results, **kwargs):
            ...         # Process the evaluation results to compute performance metrics
            ...         successful_count = sum(1 for r in grader_results if r.success)
            ...         return AnalysisResult(
            ...             name=self.name,
            ...             metadata={
            ...                 "total_samples": len(data),
            ...                 "successful_completions": successful_count,
            ...                 "success_rate": successful_count / len(data) if data else 0
            ...             }
            ...         )
            >>> analyzer = PerformanceAnalyzer()
            >>> # Note: This is a simplified example - actual implementation would
            >>> # process the grader_results more thoroughly
        """
