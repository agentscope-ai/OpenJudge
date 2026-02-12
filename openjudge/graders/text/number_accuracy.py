"""Number Accuracy Grader Module.

This module implements a grader that evaluates the numerical accuracy of response content
by comparing numbers extracted from the response text with those in a reference response text.
It is particularly useful for evaluating mathematical computations, data reporting,
and other content where numeric precision is important.

The NumberAccuracyGrader class identifies numerical values in both texts and compares
them with a configurable tolerance to determine accuracy scores.
"""

import re
from typing import Any, List

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class NumberAccuracyGrader(BaseGrader):
    """
    Check numerical calculation accuracy by comparing numbers in response vs reference response content.

    This reward verifies if the numbers in the response content match
    the numbers in the reference_response content within a specified tolerance.

    Methods:
        evaluate: Extracts and compares numbers between response and reference_response content.

    Examples:
        >>> grader = NumberAccuracyGrader(tolerance=1e-6)
        >>> result = await grader.aevaluate(
        ...     response="The result is 3.14159",
        ...     reference_response="The result is 3.14159"
        ... )
        >>> print(result.score)
        1.0
        >>> result = await grader.aevaluate(
        ...     response="The result is 3.14",
        ...     reference_response="The result is 3.14159"
        ... )
        >>> print(result.score)
        0.0
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize NumberAccuracyGrader.

        Args:
            tolerance: Tolerance for number comparison. Default is 1e-6.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional keyword arguments passed to BaseGrader.

        Example:
            >>> grader = NumberAccuracyGrader(tolerance=0.01)
        """
        super().__init__(
            name="number_accuracy",
            mode=GraderMode.POINTWISE,
            description="Check numerical calculation accuracy by comparing numbers in response "
            "vs reference_response content",
            strategy=strategy,
            **kwargs,
        )
        self.tolerance = tolerance
        self._number_pattern = re.compile(r"-?\d+\.?\d*")

    def _extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text.

        Args:
            text: Input text to extract numbers from.

        Returns:
            List[float]: List of extracted numbers as floats.

        Example:
            >>> grader = NumberAccuracyGrader()
            >>> grader._extract_numbers("The result is 3.14 and 42")
            [3.14, 42.0]
        """
        numbers = self._number_pattern.findall(text)
        return [float(n) for n in numbers if n]

    async def _aevaluate(self, response: str, reference_response: str) -> GraderScore:
        """
        Calculate number accuracy by comparing extracted numbers from both texts.

        This method extracts numerical values from both the response and reference_response texts,
        then compares them in order to compute an accuracy score. The score represents the
        proportion of numbers in the reference_response that were correctly reproduced in the response text.
        Numbers are compared with a tolerance threshold to account for floating-point precision issues.

        Args:
            response (str): Generated content to evaluate. This is typically the output
                from a language model that we want to assess for numerical accuracy.
            reference_response (str): Reference answer containing expected numbers. The numbers
                in this text are considered the reference response.

        Returns:
            GraderScore: Result containing the number accuracy score and explanation.
                - score (float): Proportion of correctly matched numbers (between 0.0 and 1.0)
                - reason (str): Explanation of the scoring result including count of correct numbers
                - metadata (Dict): Contains details about extracted numbers and comparison including:
                    - accuracy (float): Computed accuracy score
                    - correct_numbers (int): Count of correctly matched numbers
                    - total_reference_response_numbers (int): Total count of numbers in reference_response text
                    - response_numbers (List[float]): Numbers extracted from response text
                    - reference_response_numbers (List[float]): Numbers extracted from reference_response text
                    - tolerance (float): Tolerance used for number comparison

        Examples:
            >>> grader = NumberAccuracyGrader(tolerance=1e-6)
            >>> result = await grader.aevaluate(
            ...     response="The result is 3.14159",
            ...     reference_response="The result is 3.14159"
            ... )
            >>> print(result.score)
            1.0
            >>> print(result.reason)
            Number accuracy: 1/1 numbers correct
            >>> result = await grader.aevaluate(
            ...     response="The temperatures are 25.5 and 30.2 degrees",
            ...     reference_response="The temperatures are 25.5 and 30.0 degrees"
            ... )
            >>> print(result.score)
            0.5
            >>> print(result.reason)
            Number accuracy: 1/2 numbers correct
        """
        response_numbers = self._extract_numbers(response)
        reference_response_numbers = self._extract_numbers(reference_response)

        if not reference_response_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No reference_response numbers to compare",
                metadata={
                    "response_numbers": response_numbers,
                    "reference_response_numbers": reference_response_numbers,
                },
            )
        if not response_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No numbers found in response content",
                metadata={
                    "response_numbers": response_numbers,
                    "reference_response_numbers": reference_response_numbers,
                },
            )

        # Compare numbers (match in order)
        correct = 0
        total = min(len(response_numbers), len(reference_response_numbers))

        for i in range(total):
            if abs(response_numbers[i] - reference_response_numbers[i]) <= self.tolerance:
                correct += 1

        accuracy = correct / len(reference_response_numbers) if reference_response_numbers else 0.0

        return GraderScore(
            name=self.name,
            score=accuracy,
            reason=f"Number accuracy: {correct}/{len(reference_response_numbers)} numbers correct",
            metadata={
                "accuracy": accuracy,
                "correct_numbers": correct,
                "total_reference_response_numbers": len(reference_response_numbers),
                "response_numbers": response_numbers,
                "reference_response_numbers": reference_response_numbers,
                "tolerance": self.tolerance,
            },
        )
