# -*- coding: utf-8 -*-
"""False Positive analyzer for computing false positive rates of graders.

This module provides an analyzer for computing the false positive rate of graders
by comparing their results against ground truth labels in the data.
"""

from typing import List

from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class FalsePositiveAnalysisResult(AnalysisResult):
    """Result of false positive analysis for a grader.

    This class contains the computed false positive rate for a grader.

    Attributes:
        false_positive_rate (float): The computed false positive rate.

    Example:
        >>> result = FalsePositiveAnalysisResult(
        ...     name="test_grader",
        ...     false_positive_rate=0.15,
        ...     metadata={"explanation": "15% of negative samples were incorrectly classified as positive"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.false_positive_rate)
        0.15
    """

    false_positive_rate: float = Field(
        default=0.0,
        description="The computed false positive rate",
    )


class FalsePositiveAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing false positive rates of graders.

    This analyzer computes the false positive rate of a grader by comparing their results
    against label outcomes in the data. It calculates the ratio of negative samples
    incorrectly classified as positive.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information that can be evaluated
    as correct or incorrect.

    Attributes:
        name (str): Name of the analyzer, defaults to "False Positive Analysis".
        prediction_threshold (float): Threshold for converting scores to binary predictions.
                                    Defaults to 0.5.

    Example:
        >>> analyzer = FalsePositiveAnalyzer(prediction_threshold=0.7)
        >>> print(analyzer.name)
        False Positive Analysis
        >>> print(analyzer.prediction_threshold)
        0.7
    """

    name: str = "False Positive Analysis"
    prediction_threshold: float = Field(
        default=0.5,
        description="Threshold for converting scores to binary predictions",
    )

    def __init__(self, prediction_threshold: float = 0.5, **data):
        """Initialize the FalsePositiveAnalyzer.

        Args:
            prediction_threshold: Threshold for converting scores to binary predictions.
                               Defaults to 0.5.
            **data: Additional data to pass to the parent class.

        Example:
            >>> analyzer = FalsePositiveAnalyzer(prediction_threshold=0.8)
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
    ) -> FalsePositiveAnalysisResult:
        """Compute the false positive rate of a grader based on evaluation results.

        Calculates the false positive rate for a grader by comparing their predictions
        with the label ground truth values in the data. False positive rate is defined as
        the proportion of negative samples incorrectly classified as positive.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            FalsePositiveAnalysisResult: The computed false positive analysis result containing
            false positive rate and metadata with explanation.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> dataset = [
            ...     {"input": "query1", "label": 0},
            ...     {"input": "query2", "label": 1}
            ... ]
            >>> grader_results = [
            ...     GraderResult(name="grader1", score=0.8, reason="Positive"),
            ...     GraderResult(name="grader1", score=0.3, reason="Negative")
            ... ]
            >>> analyzer = FalsePositiveAnalyzer(prediction_threshold=0.5)
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            False Positive Analysis
            >>> print(f"False Positive Rate: {result.false_positive_rate:.2f}")
            False Positive Rate: 1.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for false positive rate calculation",
            )
            return FalsePositiveAnalysisResult(
                name=self.name,
                false_positive_rate=0.0,
                metadata={
                    "explanation": "No data or grader results provided for false positive rate calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)
        valid_results = 0

        # Counters for false positives and true negatives
        false_positives = 0
        true_negatives = 0
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
                if prediction == 1 and label == 0:
                    false_positives += 1
                elif prediction == 0 and label == 0:
                    true_negatives += 1

        # Calculate false positive rate
        total_negatives = false_positives + true_negatives
        if total_negatives == 0:
            fp_rate = 0.0
            explanation = "No negative samples found for false positive rate calculation"
        else:
            fp_rate = false_positives / total_negatives
            explanation = (
                f"False Positives: {false_positives}, True Negatives: {true_negatives}, "
                f"False Positive Rate: {fp_rate:.2%}"
            )

        return FalsePositiveAnalysisResult(
            name=self.name,
            false_positive_rate=fp_rate,
            metadata={
                **self._coverage_metadata(grader_results, valid_results, explanation),
                "false_positives": false_positives,
                "true_negatives": true_negatives,
            },
        )
