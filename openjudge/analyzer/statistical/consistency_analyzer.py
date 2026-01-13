# -*- coding: utf-8 -*-
"""Consistency analyzer for computing consistency scores between repeated evaluations.

This module provides an analyzer for computing the consistency of grader evaluations
by comparing results from repeated runs of the same grader on identical inputs.
"""

from typing import List

import numpy as np
from loguru import logger
from pydantic import Field

from openjudge.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from openjudge.graders.schema import GraderResult, GraderScore


class ConsistencyAnalysisResult(AnalysisResult):
    """Result of consistency analysis for a grader.

    This class contains the computed consistency score for a grader.

    Attributes:
        consistency (float): The computed consistency score (correlation between repeated evaluations).

    Example:
        >>> result = ConsistencyAnalysisResult(
        ...     name="test_grader",
        ...     consistency=0.95,
        ...     metadata={"explanation": "High consistency between repeated evaluations"}
        ... )
        >>> print(result.name)
        test_grader
        >>> print(result.consistency)
        0.95
    """

    consistency: float = Field(
        default=0.0,
        description="The computed consistency score",
    )


class ConsistencyAnalyzer(BaseAnalyzer):
    """Analyzer for computing consistency scores of graders.

    This analyzer computes the consistency of a grader by comparing results from
    repeated evaluations on identical inputs. It calculates the correlation between
    scores from different runs to measure stability.

    The analyzer expects paired results from repeated evaluations of the same samples.

    Attributes:
        name (str): Name of the analyzer, defaults to "Consistency Analysis".

    Example:
        >>> analyzer = ConsistencyAnalyzer()
        >>> print(analyzer.name)
        Consistency Analysis
    """

    name: str = "Consistency Analysis"

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        another_grader_results: List[GraderResult],
        **kwargs,
    ) -> ConsistencyAnalysisResult:
        """Compute the consistency of a grader based on repeated evaluation results.

        Calculates the consistency score for a grader by comparing scores from two
        separate evaluations of the same inputs. Consistency is measured as the
        correlation between the two sets of scores.

        Args:
            dataset: The collection of data samples that were evaluated.
            grader_results: The first evaluation results from graders.
            another_grader_results: The second evaluation results from graders
            **kwargs: Additional keyword arguments.

        Returns:
            ConsistencyAnalysisResult: The computed consistency analysis result containing
            consistency score and metadata with explanation.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> first_run = [
            ...     GraderResult(name="grader1", score=0.8, reason="High quality"),
            ...     GraderResult(name="grader1", score=0.3, reason="Low quality")
            ... ]
            >>> second_run = [
            ...     GraderResult(name="grader1", score=0.75, reason="High quality"),
            ...     GraderResult(name="grader1", score=0.35, reason="Low quality")
            ... ]
            >>> analyzer = ConsistencyAnalyzer()
            >>> result = analyzer.analyze(first_run, second_run)
            >>> print(result.name)
            Consistency Analysis
            >>> print(f"Consistency: {result.consistency:.2f}")
            Consistency: 0.99
        """
        # Need to support old 2-argment call signagure: analyze(first_run_results, second_run_results)
        # Need to determine which argment is the 1st run result and which is the 2nd run result.
        if grader_results and another_grader_results:
            # current call signature
            first_run_results = grader_results
            second_run_results = another_grader_results
        elif dataset and grader_results:
            # The first two argments contain values but the 3rd does not.
            # Treat this as a call following the old 2-argument signature.
            first_run_results = dataset
            second_run_results = grader_results
        else:
            # 1. Insufficient argments for the current call signature:
            # dataset and another grader result exist,
            # but the 2nd argment (grader result) does not have value.
            # Or 2. none of dataset, grader_results, another_grader_results exists.
            first_run_results = []
            second_run_results = []

        if not first_run_results or not second_run_results:
            logger.warning(
                "No results provided for consistency calculation",
            )
            return ConsistencyAnalysisResult(
                name=self.name,
                consistency=0.0,
                metadata={
                    "explanation": "No results provided for consistency calculation",
                },
            )

        # Collect scores from both runs
        first_run_scores = []
        second_run_scores = []

        # Iterate over paired results and extract scores
        for first_result, second_result in zip(first_run_results, second_run_results):
            if not first_result or not second_result:
                continue

            if (
                isinstance(first_result, GraderScore)
                and hasattr(first_result, "score")
                and isinstance(second_result, GraderScore)
                and hasattr(second_result, "score")
            ):
                first_run_scores.append(first_result.score)
                second_run_scores.append(second_result.score)

        # Calculate consistency as correlation between the two runs
        if len(first_run_scores) < 2:
            consistency_score = 0.0
            explanation = "Insufficient data points for consistency calculation"
        else:
            try:
                # Calculate Pearson correlation coefficient
                correlation_matrix = np.corrcoef(first_run_scores, second_run_scores)
                consistency_score = correlation_matrix[0, 1]

                # Handle NaN case - occurs when one array has zero variance (all values identical)
                if np.isnan(consistency_score):
                    # If all values in either array are identical, perfect consistency if both arrays
                    # have constant values that match each other
                    first_unique = len(set(first_run_scores)) == 1
                    second_unique = len(set(second_run_scores)) == 1
                    if first_unique and second_unique and first_run_scores[0] == second_run_scores[0]:
                        consistency_score = 1.0
                    else:
                        # If one array has variance and the other doesn't, they're inconsistent
                        consistency_score = 0.0

                explanation = (
                    f"Consistency based on {len(first_run_scores)} paired evaluations: {consistency_score:.4f}"
                )
            except Exception as e:
                consistency_score = 0.0
                explanation = f"Error calculating consistency: {str(e)}"

        return ConsistencyAnalysisResult(
            name=self.name,
            consistency=consistency_score,
            metadata={
                "explanation": explanation,
                "paired_evaluations": len(first_run_scores),
            },
        )
