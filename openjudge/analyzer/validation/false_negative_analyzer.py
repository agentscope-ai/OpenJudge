# -*- coding: utf-8 -*-
"""False Negative analyzer for computing false negative rates of graders.

This module provides an analyzer for computing the false negative rate of graders
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


class FalseNegativeAnalysisResult(AnalysisResult):
    """Result of false negative analysis for a grader.

    This class contains the computed false negative rate for a grader.

    Attributes:
        false_negative_rate (float): The computed false negative rate.

    Example:
        >>> result = FalseNegativeAnalysisResult(
        ...     name="test_grader",
        ...     false_negative_rate=0.15,
        ...     metadata={"explanation": "15% of positive samples were incorrectly classified as negative"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.false_negative_rate)
        0.15
    """

    false_negative_rate: float = Field(
        default=0.0,
        description="The computed false negative rate",
    )


class FalseNegativeAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing false negative rates of graders.

    This analyzer computes the false negative rate of a grader by comparing their results
    against label outcomes in the data. It calculates the ratio of positive samples
    incorrectly classified as negative.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information that can be evaluated
    as correct or incorrect.

    Attributes:
        name (str): Name of the analyzer, defaults to "False Negative Analysis".
        prediction_threshold (float): Threshold for converting scores to binary predictions.
                                    Defaults to 0.5.

    Example:
        >>> analyzer = FalseNegativeAnalyzer(prediction_threshold=0.7)
        >>> print(analyzer.name)
        False Negative Analysis
        >>> print(analyzer.prediction_threshold)
        0.7
    """

    name: str = "False Negative Analysis"
    prediction_threshold: float = Field(
        default=0.5,
        description="Threshold for converting scores to binary predictions",
    )

    def __init__(self, prediction_threshold: float = 0.5, **data):
        """Initialize the FalseNegativeAnalyzer.

        Args:
            prediction_threshold: Threshold for converting scores to binary predictions.
                               Defaults to 0.5.
            **data: Additional data to pass to the parent class.

        Example:
            >>> analyzer = FalseNegativeAnalyzer(prediction_threshold=0.8)
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
    ) -> FalseNegativeAnalysisResult:
        """Compute the false negative rate of a grader based on evaluation results.

        Calculates the false negative rate for a grader by comparing their predictions
        with the label ground truth values in the data. False negative rate is defined as
        the proportion of positive samples incorrectly classified as negative.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            FalseNegativeAnalysisResult: The computed false negative analysis result containing
            false negative rate and metadata with explanation.

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
            >>> analyzer = FalseNegativeAnalyzer(prediction_threshold=0.5)
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            False Negative Analysis
            >>> print(f"False Negative Rate: {result.false_negative_rate:.2f}")
            False Negative Rate: 0.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for false negative rate calculation",
            )
            return FalseNegativeAnalysisResult(
                name=self.name,
                false_negative_rate=0.0,
                metadata={
                    "explanation": "No data or grader results provided for false negative rate calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)
        valid_results = 0

        # Counters for false negatives and true positives
        false_negatives = 0
        true_positives = 0
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
                if prediction == 0 and label == 1:
                    false_negatives += 1
                elif prediction == 1 and label == 1:
                    true_positives += 1

        # Calculate false negative rate
        total_positives = false_negatives + true_positives
        if total_positives == 0:
            fn_rate = 0.0
            explanation = "No positive samples found for false negative rate calculation"
        else:
            fn_rate = false_negatives / total_positives
            explanation = (
                f"False Negatives: {false_negatives}, True Positives: {true_positives}, "
                f"False Negative Rate: {fn_rate:.2%}"
            )

        return FalseNegativeAnalysisResult(
            name=self.name,
            false_negative_rate=fn_rate,
            metadata={
                **self._coverage_metadata(grader_results, valid_results, explanation),
                "false_negatives": false_negatives,
                "true_positives": true_positives,
            },
        )
