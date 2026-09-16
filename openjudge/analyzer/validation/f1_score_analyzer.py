# -*- coding: utf-8 -*-
"""F1 score analyzer for computing precision, recall, and F1 scores of graders.

This module provides an analyzer for computing the F1 score of graders by comparing
their results against ground truth label sets in the data. It calculates precision,
recall, and F1 score for cases where both predictions and ground truth are sets of items.
"""

from typing import List

from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.graders.schema import GraderResult, GraderScore


class F1ScoreAnalysisResult(AnalysisResult):
    """Result of F1 score analysis for a grader.

    This class contains the computed precision, recall, and F1 score for a grader.

    Attributes:
        precision (float): The computed precision score.
        recall (float): The computed recall score.
        f1_score (float): The computed F1 score.

    Example:
        >>> result = F1ScoreAnalysisResult(
        ...     name="test_grader",
        ...     precision=0.8,
        ...     recall=0.7,
        ...     f1_score=0.75
        ... )
        >>> print(result.name)
        test_grader
        >>> print(f"F1 Score: {result.f1_score}")
        F1 Score: 0.75
    """

    precision: float = Field(
        default=0.0,
        description="The computed precision score",
    )
    recall: float = Field(
        default=0.0,
        description="The computed recall score",
    )
    f1_score: float = Field(
        default=0.0,
        description="The computed F1 score",
    )


class F1ScoreAnalyzer(BaseValidationAnalyzer):
    """Analyzer for computing F1 scores of graders.

    This analyzer computes the F1 score of a grader by comparing their results
    against label sets in the data. It calculates precision, recall, and F1 score
    for cases where both predictions and ground truth are sets of items.

    The analyzer expects the ground truth to be present in the data samples as sets
    and the grader results to contain set information that can be evaluated.

    Attributes:
        name (str): Name of the analyzer, defaults to "F1 Score Analysis".
        prediction_threshold (float): Threshold for converting scores to binary predictions.
                                    Defaults to 0.5.

    Example:
        >>> analyzer = F1ScoreAnalyzer(prediction_threshold=0.7)
        >>> print(analyzer.name)
        F1 Score Analysis
        >>> print(analyzer.prediction_threshold)
        0.7
    """

    name: str = "F1 Score Analysis"
    prediction_threshold: float = Field(
        default=0.5,
        description="Threshold for converting scores to binary predictions",
    )

    def __init__(self, prediction_threshold: float = 0.5, **data):
        """Initialize the F1ScoreAnalyzer.

        Args:
            prediction_threshold: Threshold for converting scores to binary predictions.
                               Defaults to 0.5.
            **data: Additional data to pass to the parent class.

        Example:
            >>> analyzer = F1ScoreAnalyzer(prediction_threshold=0.8)
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
    ) -> F1ScoreAnalysisResult:
        """Compute the F1 score of a grader based on evaluation results.

        Calculates precision, recall, and F1 score for a grader by comparing their
        predicted sets with the ground truth sets in the data.

        Args:
            dataset: The data samples that were evaluated. Each dict represents one sample
                with its input parameters and ground truth sets.
            grader_results: The evaluation results from a single
                grader, organized as a list of GraderResult objects, one for each sample.
            label_path: The key or path to extract the ground truth label set from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answers".
            **kwargs: Additional keyword arguments.

        Returns:
            F1ScoreAnalysisResult: The computed F1 score analysis result containing
            precision, recall, and F1 score with metadata.

        Example:
            >>> from openjudge.graders.schema import GraderResult
            >>> dataset = [
            ...     {"input": "query1", "label": {"a", "b"}},
            ...     {"input": "query2", "label": {"c"}}
            ... ]
            >>> grader_results = [
            ...     GraderResult(name="grader1", metadata={"predictions": {"a", "c"}}, reason="Good"),
            ...     GraderResult(name="grader1", metadata={"predictions": {"c", "d"}}, reason="Acceptable")
            ... ]
            >>> analyzer = F1ScoreAnalyzer(prediction_threshold=0.7)
            >>> result = analyzer.analyze(dataset, grader_results)
            >>> print(f"F1 Score: {result.f1_score:.3f}")
            F1 Score: 0.667
        """
        if not dataset or not grader_results:
            logger.warning(
                "No data or grader results provided for F1 score calculation",
            )
            return F1ScoreAnalysisResult(
                name=self.name,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                metadata={
                    "explanation": "No data or grader results provided for F1 score calculation",
                },
            )

        self._validate_alignment(dataset, grader_results)
        valid_results = 0

        # Counters for TP, FP, FN
        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives

        # Iterate over each sample and compare grader results with label values
        for sample, grader_result in zip(dataset, grader_results):
            label_set = self._extract(sample, label_path)
            if label_set is None:
                continue

            if not grader_result:
                continue
            valid_results += 1

            # Extract predicted set from grader result
            predicted_set = set()
            if hasattr(grader_result, "metadata") and "predictions" in grader_result.metadata:
                predicted_set = set(grader_result.metadata["predictions"])
            elif isinstance(grader_result, GraderScore) and hasattr(
                grader_result,
                "score",
            ):
                # For simple case where score represents a single item prediction
                if grader_result.score >= self.prediction_threshold:
                    predicted_set = {str(grader_result.score)}

            # Convert label to set if it's not already
            if not isinstance(label_set, (set, list, tuple)):
                label_set = {label_set}

            if not isinstance(label_set, set):
                label_set = set(label_set)

            # Calculate TP, FP, FN for this sample
            tp = len(predicted_set.intersection(label_set))  # True positives
            fp = len(predicted_set.difference(label_set))  # False positives
            fn = len(label_set.difference(predicted_set))  # False negatives

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calculate precision, recall, and F1 score
        if total_tp + total_fp == 0:
            precision = 0.0
        else:
            precision = total_tp / (total_tp + total_fp)

        if total_tp + total_fn == 0:
            recall = 0.0
        else:
            recall = total_tp / (total_tp + total_fn)

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        explanation = (
            f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, "
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}"
        )

        return F1ScoreAnalysisResult(
            name=self.name,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            metadata={
                **self._coverage_metadata(grader_results, valid_results, explanation),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
            },
        )
