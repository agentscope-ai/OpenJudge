# -*- coding: utf-8 -*-
"""Precision analyzer for computing precision scores of graders.

This module provides an analyzer for computing the precision of graders by comparing
their results against ground truth labels in the data. It calculates precision
as the ratio of true positive predictions to the total positive predictions.
"""

from typing import List

from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class PrecisionAnalysisResult(AnalysisResult):
    """Result of precision analysis for a grader.

    This class contains the computed precision score for a grader.

    Attributes:
        precision (float): The computed precision score.

    Example:
        >>> result = PrecisionAnalysisResult(
        ...     name="test_grader",
        ...     precision=0.85,
        ...     metadata={"explanation": "85% of positive predictions were correct"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.precision)
        0.85
    """

    precision: float = Field(
        default=0.0,
        description="The computed precision score",
    )


class PrecisionAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing precision scores of graders.

    This analyzer computes the precision of a grader by comparing their results
    against label outcomes in the data. It calculates the ratio of true positive
    predictions to the total positive predictions.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information that can be evaluated
    as correct or incorrect.

    Attributes:
        name (str): Name of the analyzer, defaults to "Precision Analysis".
        prediction_threshold (float): Threshold for converting scores to binary predictions.
                                    Defaults to 0.5.

    Example:
        >>> analyzer = PrecisionAnalyzer(prediction_threshold=0.7)
        >>> print(analyzer.name)
        Precision Analysis
        >>> print(analyzer.prediction_threshold)
        0.7
    """

    name: str = "Precision Analysis"
    prediction_threshold: float = Field(
        default=0.5,
        description="Threshold for converting scores to binary predictions",
    )

    def __init__(self, prediction_threshold: float = 0.5, **data):
        """Initialize the PrecisionAnalyzer.

        Args:
            prediction_threshold: Threshold for converting scores to binary predictions.
                               Defaults to 0.5.
            **data: Additional data to pass to the parent class.

        Example:
            >>> analyzer = PrecisionAnalyzer(prediction_threshold=0.8)
            >>> print(analyzer.prediction_threshold)
            0.8
        """
        super().__init__(**data)
        self.prediction_threshold = prediction_threshold

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        label_path: str = "label",
        **kwargs,
    ) -> PrecisionAnalysisResult:
        """Compute the precision of a grader based on evaluation results.

        Calculates the precision score for a grader by comparing their predictions
        with the label ground truth values in the data. Precision is defined as
        the proportion of true positive predictions among all positive predictions.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            PrecisionAnalysisResult: The computed precision analysis result containing
            precision score and metadata with explanation.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> dataset = [
            ...     {"input": "query1", "label": 1},
            ...     {"input": "query2", "label": 0}
            ... ]
            >>> grader_results = [
            ...     GraderResult(name="grader1", score=0.8, reason="Positive"),
            ...     GraderResult(name="grader1", score=0.3, reason="Negative")
            ... ]
            >>> analyzer = PrecisionAnalyzer(prediction_threshold=0.5)
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            Precision Analysis
            >>> print(f"Precision: {result.precision:.2f}")
            Precision: 1.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for precision calculation",
            )
            return PrecisionAnalysisResult(
                name=self.name,
                precision=0.0,
                metadata={
                    "explanation": "No data or grader results provided for precision calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)
        valid_results = 0

        # Counters for true positives and false positives
        true_positives = 0
        false_positives = 0
        # Iterate over each sample and compare grader results with label values
        for sample, grader_result in zip(dataset, grader_results):
            label = self._extract(sample, label_path)
            if label is None:
                continue

            if not grader_result:
                continue

            if isinstance(grader_result, GraderScore) and hasattr(grader_result, "score"):
                valid_results += 1
                predicted_value = grader_result.score
                prediction = 1 if predicted_value >= self.prediction_threshold else 0

                # Compare prediction with label value
                if prediction == 1 and label == 1:
                    true_positives += 1
                elif prediction == 1 and label == 0:
                    false_positives += 1

        # Calculate precision
        total_positive_predictions = true_positives + false_positives
        if total_positive_predictions == 0:
            precision_score = 0.0
            explanation = "No positive predictions found for precision calculation"
        else:
            precision_score = true_positives / total_positive_predictions
            explanation = (
                f"True Positives: {true_positives}, False Positives: {false_positives}, "
                f"Precision: {precision_score:.2%}"
            )

        return PrecisionAnalysisResult(
            name=self.name,
            precision=precision_score,
            metadata={
                **self._coverage_metadata(grader_results, valid_results, explanation),
                "true_positives": true_positives,
                "false_positives": false_positives,
            },
        )
