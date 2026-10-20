# -*- coding: utf-8 -*-
"""Correlation analyzer for computing correlation between grader scores and ground truth.

This module provides an analyzer for computing the correlation between grader scores
and ground truth labels in the data. It calculates Pearson correlation coefficient
to measure the linear relationship between predicted scores and actual labels.
"""

from typing import List

import numpy as np
from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class CorrelationAnalysisResult(AnalysisResult):
    """Result of correlation analysis for a grader.

    This class contains the computed correlation score for a grader.

    Attributes:
        correlation (float): The computed correlation score.

    Example:
        >>> result = CorrelationAnalysisResult(
        ...     name="test_grader",
        ...     correlation=0.85,
        ...     metadata={"explanation": "Strong positive correlation between predictions and labels"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.correlation)
        0.85
    """

    correlation: float = Field(
        default=0.0,
        description="The computed correlation score",
    )


class CorrelationAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing correlation scores of graders.

    This analyzer computes the correlation of a grader by comparing their results
    against label outcomes in the data. It calculates the Pearson correlation coefficient
    between predicted scores and actual labels.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information.

    Attributes:
        name (str): Name of the analyzer, defaults to "Correlation Analysis".

    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> print(analyzer.name)
        Correlation Analysis
    """

    name: str = "Correlation Analysis"

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        label_path: str = "label",
        **kwargs,
    ) -> CorrelationAnalysisResult:
        """Compute the correlation of a grader based on evaluation results.

        Calculates the correlation score for a grader by comparing their predictions
        with the label ground truth values in the data. Correlation is defined as
        the Pearson correlation coefficient between predicted scores and actual labels.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            CorrelationAnalysisResult: The computed correlation analysis result containing
            correlation score and metadata with explanation.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> dataset = [
            ...     {"input": "query1", "label": 0.8},
            ...     {"input": "query2", "label": 0.2}
            ... ]
            >>> grader_results = [
            ...     GraderResult(name="grader1", score=0.7, reason="High quality"),
            ...     GraderResult(name="grader1", score=0.3, reason="Low quality")
            ... ]
            >>> analyzer = CorrelationAnalyzer()
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            Correlation Analysis
            >>> print(f"Correlation: {result.correlation:.2f}")
            Correlation: 1.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for correlation calculation",
            )
            return CorrelationAnalysisResult(
                name=self.name,
                correlation=0.0,
                metadata={
                    "explanation": "No data or grader results provided for correlation calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)

        # Collect predicted scores and ground truth labels
        predicted_scores = []
        ground_truth_labels = []

        # Iterate over each sample and compare grader results with label values
        for sample, grader_result in zip(dataset, grader_results):
            label = self._extract(sample, label_path)
            if label is None:
                continue

            if not grader_result:
                continue

            if isinstance(grader_result, GraderScore) and hasattr(grader_result, "score"):
                predicted_scores.append(grader_result.score)
                ground_truth_labels.append(label)

        # Calculate correlation
        if len(predicted_scores) < 2:
            correlation_score = 0.0
            explanation = "Insufficient data points for correlation calculation"
        else:
            try:
                # Calculate Pearson correlation coefficient
                correlation_matrix = np.corrcoef(predicted_scores, ground_truth_labels)
                correlation_score = correlation_matrix[0, 1]

                # Handle NaN case - occurs when one array has zero variance (all values identical)
                if np.isnan(correlation_score):
                    # If all values in either array are identical, perfect correlation if both
                    # arrays have constant values that match each other
                    scores_unique = len(set(predicted_scores)) == 1
                    labels_unique = len(set(ground_truth_labels)) == 1
                    if scores_unique and labels_unique and predicted_scores[0] == ground_truth_labels[0]:
                        correlation_score = 1.0
                    else:
                        # If one array has variance and the other doesn't, they're uncorrelated
                        correlation_score = 0.0

                explanation = f"Correlation based on {len(predicted_scores)} data points: {correlation_score:.4f}"
            except Exception as e:
                correlation_score = 0.0
                explanation = f"Error calculating correlation: {str(e)}"

        return CorrelationAnalysisResult(
            name=self.name,
            correlation=correlation_score,
            metadata={
                **self._coverage_metadata(grader_results, len(predicted_scores), explanation),
                "data_points": len(predicted_scores),
            },
        )
