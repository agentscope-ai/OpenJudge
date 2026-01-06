# -*- coding: utf-8 -*-
"""
Accuracy-specific grader validation functionality.

This module provides specialized validation capabilities for graders
that measure accuracy by comparing predictions with expected outcomes.
"""

from typing import Dict, List

from cookbooks.grader_validation.base import GraderValidator
from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.accuracy_analyzer import AccuracyAnalyzer
from openjudge.graders.base_grader import BaseGrader
from openjudge.runner.grading_runner import GradingRunner


class AccuracyGraderValidator(GraderValidator):
    """
    Validator specifically for accuracy-based graders.

    This class specializes the generic GraderValidator for cases where
    accuracy analysis is required. It ensures that the analyzer used
    is specifically an AccuracyAnalyzer.

    Attributes:
        analyzer (AccuracyAnalyzer): Analyzer specialized for computing accuracy metrics.
    """

    def __init__(self, runner: GradingRunner):
        super().__init__(runner, analyzer=AccuracyAnalyzer())

    async def validate(
        self,
        dataset: List[dict],
        grader: BaseGrader,
        mapping: Dict[str, str],
    ) -> AnalysisResult:
        """
        Validate a grader's accuracy by running it on test data and analyzing accuracy metrics.

        This method executes the grader on the provided data samples using the
        specified field mappings, then analyzes the accuracy of results using
        the configured AccuracyAnalyzer.

        Args:
            dataset (List[dict]): List of data samples to evaluate. Each dictionary
                represents a sample with input parameters and expected outputs.
                For accuracy validation, each sample should include an "expected" key
                with the reference response value.
            grader (BaseGrader): The grader instance to validate for accuracy.
            mapping (Dict[str, str]): Field mapping dictionary that maps grader
                input parameter names to corresponding keys in the data samples.

        Returns:
            AnalysisResult: Accuracy analysis result containing metrics such as
                overall accuracy score and detailed breakdown.

        Example:
            >>> accuracy_analyzer = AccuracyAnalyzer()
            >>> validator = AccuracyGraderValidator(runner=my_runner, analyser=accuracy_analyzer)
            >>> dataset = [
            ...     {"query": "What is 2+2?", "expected": 4},
            ...     {"query": "Capital of France?", "expected": "Paris"}
            ... ]
            >>> my_grader = MyMathGrader()
            >>> mapping = {"question": "query"}
            >>> result = await validator.validate(dataset, my_grader, mapping)
            >>> print(f"Accuracy: {result.accuracy}")
        """
        return await super().validate(dataset, grader, mapping)
