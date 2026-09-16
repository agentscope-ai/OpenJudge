# -*- coding: utf-8 -*-
"""Accuracy analyzer for computing accuracy scores of graders.

This module provides an analyzer for computing the accuracy of graders by comparing
their results against ground truth labels in the data.
"""

from typing import List

from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class AccuracyAnalysisResult(AnalysisResult):
    """Result of accuracy analysis for a grader.

    This class contains the computed accuracy score for a grader.

    Attributes:
        accuracy (float): The computed accuracy score.

    Example:
        >>> result = AccuracyAnalysisResult(
        ...     name="test_grader",
        ...     accuracy=0.85,
        ...     metadata={"explanation": "Correctly predicted 85 out of 100 samples"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.accuracy)
        0.85
    """

    accuracy: float = Field(
        default=0.0,
        description="The computed accuracy score",
    )


class AccuracyAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing accuracy scores of graders.

    This analyzer computes the accuracy of a grader by comparing their results
    against label outcomes in the data. It calculates the ratio of correct
    predictions made by the grader.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information that can be evaluated
    as correct or incorrect.

    Attributes:
        name (str): Name of the analyzer, defaults to "Accuracy Analysis".

    Example:
        >>> analyzer = AccuracyAnalyzer()
        >>> print(analyzer.name)
        Accuracy Analysis
    """

    name: str = "Accuracy Analysis"

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        label_path: str = "label",
        **kwargs,
    ) -> AccuracyAnalysisResult:
        """Compute the accuracy of a grader based on evaluation results.

        Calculates the accuracy score for a grader by comparing their predictions
        with the label ground truth values in the data. The accuracy is defined
        as the proportion of correct predictions among all predictions.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            AccuracyAnalysisResult: The computed accuracy analysis result containing
            accuracy score and metadata with explanation.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> dataset = [
            ...     {"input": "query1", "label": 1},
            ...     {"input": "query2", "label": 0}
            ... ]
            >>> grader_results = [
            ...     GraderResult(name="grader1", score=1.0, reason="Correct"),
            ...     GraderResult(name="grader1", score=0.0, reason="Incorrect")
            ... ]
            >>> analyzer = AccuracyAnalyzer()
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            Accuracy Analysis
            >>> print(f"Accuracy: {result.accuracy:.2f}")
            Accuracy: 1.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for accuracy calculation",
            )
            return AccuracyAnalysisResult(
                name=self.name,
                accuracy=0.0,
                metadata={
                    "explanation": "No data or grader results provided for accuracy calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)

        # Counters for correct predictions and total predictions
        correct_predictions = 0
        total_predictions = 0

        # Iterate over each sample and compare grader results with label values
        for sample, grader_result in zip(dataset, grader_results):
            label = self._extract(sample, label_path)
            if label is None:
                continue

            if not grader_result:
                continue

            if isinstance(grader_result, GraderScore) and hasattr(grader_result, "score"):
                predicted_value = grader_result.score

                # Compare prediction with label value
                if predicted_value == label:
                    correct_predictions += 1
                total_predictions += 1

        # Calculate accuracy
        if total_predictions == 0:
            accuracy_score = 0.0
            explanation = "No valid predictions found for accuracy calculation"
        else:
            accuracy_score = correct_predictions / total_predictions
            explanation = (
                f"Correctly predicted {correct_predictions} out of {total_predictions} samples "
                f"({accuracy_score:.2%} accuracy)"
            )

        return AccuracyAnalysisResult(
            name=self.name,
            accuracy=accuracy_score,
            metadata={
                **self._coverage_metadata(grader_results, total_predictions, explanation),
            },
        )
