# -*- coding: utf-8 -*-
"""Recall analyzer for computing recall scores of graders.

This module provides an analyzer for computing the recall of graders by comparing
their results against ground truth labels in the data. It calculates recall
as the ratio of true positive predictions to the total actual positives.
"""

from typing import List

from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class RecallAnalysisResult(AnalysisResult):
    """Result of recall analysis for a grader.

    This class contains the computed recall score for a grader.

    Attributes:
        recall (float): The computed recall score.

    Example:
        >>> result = RecallAnalysisResult(
        ...     name="test_grader",
        ...     recall=0.85,
        ...     metadata={"explanation": "85% of actual positives were correctly identified"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.recall)
        0.85
    """

    recall: float = Field(
        default=0.0,
        description="The computed recall score",
    )


class RecallAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing recall scores of graders.

    This analyzer computes the recall of a grader by comparing their results
    against label outcomes in the data. It calculates the ratio of true positive
    predictions to the total actual positives.

    The analyzer expects the ground truth to be present in the data samples
    and the grader results to contain score information that can be evaluated
    as correct or incorrect.

    Attributes:
        name (str): Name of the analyzer, defaults to "Recall Analysis".
        prediction_threshold (float): Threshold for converting scores to binary predictions.
                                    Defaults to 0.5.

    Example:
        >>> analyzer = RecallAnalyzer(prediction_threshold=0.7)
        >>> print(analyzer.name)
        Recall Analysis
        >>> print(analyzer.prediction_threshold)
        0.7
    """

    name: str = "Recall Analysis"
    prediction_threshold: float = Field(
        default=0.5,
        description="Threshold for converting scores to binary predictions",
    )

    def __init__(self, prediction_threshold: float = 0.5, **data):
        """Initialize the RecallAnalyzer.

        Args:
            prediction_threshold: Threshold for converting scores to binary predictions.
                               Defaults to 0.5.
            **data: Additional data to pass to the parent class.

        Example:
            >>> analyzer = RecallAnalyzer(prediction_threshold=0.8)
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
    ) -> RecallAnalysisResult:
        """Compute the recall of a grader based on evaluation results.

        Calculates the recall score for a grader by comparing their predictions
        with the label ground truth values in the data. Recall is defined as
        the proportion of true positive predictions among all actual positives.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters, ground truth, and label outputs.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments.

        Returns:
            RecallAnalysisResult: The computed recall analysis result containing
            recall score and metadata with explanation.

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
            >>> analyzer = RecallAnalyzer(prediction_threshold=0.5)
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(result.name)
            Recall Analysis
            >>> print(f"Recall: {result.recall:.2f}")
            Recall: 1.00
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for recall calculation",
            )
            return RecallAnalysisResult(
                name=self.name,
                recall=0.0,
                metadata={
                    "explanation": "No data or grader results provided for recall calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)
        valid_results = 0

        # Counters for true positives and false negatives
        true_positives = 0
        false_negatives = 0
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
                elif prediction == 0 and label == 1:
                    false_negatives += 1

        # Calculate recall
        total_actual_positives = true_positives + false_negatives
        if total_actual_positives == 0:
            recall_score = 0.0
            explanation = "No actual positives found for recall calculation"
        else:
            recall_score = true_positives / total_actual_positives
            explanation = (
                f"True Positives: {true_positives}, False Negatives: {false_negatives}, " f"Recall: {recall_score:.2%}"
            )

        return RecallAnalysisResult(
            name=self.name,
            recall=recall_score,
            metadata={
                **self._coverage_metadata(grader_results, valid_results, explanation),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
            },
        )
