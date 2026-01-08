# -*- coding: utf-8 -*-
"""
Base module for grader validation functionality.

This module provides the foundational classes for validating graders
by running evaluations and analyzing the results.
"""

from abc import ABC
from typing import Dict, List

from openjudge.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from openjudge.graders.base_grader import BaseGrader
from openjudge.runner.grading_runner import GradingRunner


class GraderValidator(ABC):
    """
    Base validator for graders.

    This class provides the basic functionality to validate graders by running
    evaluations on provided data and analyzing the results using a specified analyzer.

    Attributes:
        runner (GradingRunner): Runner responsible for executing the grader on data.
        analyzer (BaseAnalyzer): Analyzer used to process and evaluate the results.
    """

    def __init__(self, runner: GradingRunner, analyzer: BaseAnalyzer) -> None:
        self.runner = runner
        self.analyzer = analyzer

    async def validate(
        self,
        dataset: List[dict],
        grader: BaseGrader,
        mapping: Dict[str, str],
    ) -> AnalysisResult:
        """
        Validate a grader by running it on test data and analyzing the results.

        This method executes the grader on the provided data samples using the
        specified field mappings, then analyzes the results using the configured analyzer.

        Args:
            dataset (List[Dict]): List of data samples to evaluate. Each dictionary
                represents a sample with input parameters and expected outputs.
            grader (BaseGrader): The grader instance to validate.
            mapping (Dict[str, str]): Field mapping dictionary that maps grader
                input parameter names to corresponding keys in the data samples.

        Returns:
            AnalysisResult: Analysis result containing metrics and evaluation information
                computed by the analyzer based on the grader's performance.

        Example:
            >>> validator = GraderValidator(runner=my_runner, analyser=my_analyzer)
            >>> dataset = [{"query": "What is 2+2?", "expected": 4}]
            >>> my_grader = MyGrader()
            >>> mapping = {"question": "query"}
            >>> result = await validator.validate(dataset, my_grader, mapping)
        """
        result = await self.runner.arun(dataset, grader, mapping)
        return self.analyzer.analyze(dataset, result["over_all"])
