"""
JSON Validator Grader Module

This module provides functionality to validate if a given text is valid JSON.
It contains the JsonValidatorGrader class which attempts to parse text as JSON
and returns a positive score if parsing succeeds.
"""

import json
from typing import Any

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class JsonValidatorGrader(BaseGrader):
    """
    JSON Format Validator Grader

    Validates if the response text is valid JSON.

    Attributes:
        name: Grader name

    Example:
        >>> grader = JsonValidatorGrader()
        >>> result = await grader.aevaluate(
        ...     reference="",  # reference not needed
        ...     response='{"valid": "json"}'
        ... )
        >>> print(f"Valid: {result.score}")
    """

    def __init__(
        self,
        name: str = "json_validator",
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the JsonValidatorGrader.

        Args:
            name (str, optional): The name of the grader. Defaults to "json_validator".
            strategy (BaseEvaluationStrategy, optional): Strategy to use for evaluation. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name=name,
            mode=GraderMode.POINTWISE,
            description="JSON format validation metric",
            strategy=strategy,
            **kwargs,
        )

    def _compute(self, response: str) -> tuple[bool, dict]:
        """
        Validate JSON format

        Returns:
            tuple[bool, dict]: (is_valid, details)
        """
        # Input validation
        if not isinstance(response, str):
            return False, {
                "is_valid": False,
                "error_message": f"Invalid input type: expected str, got {type(response).__name__}",
                "response_length": 0,
            }

        try:
            json.loads(response)
            return True, {"is_valid": True, "response_length": len(response)}
        except json.JSONDecodeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"JSON decode error: {str(e)}",
                "response_length": len(response),
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "error_message": f"Unexpected error: {str(e)}",
                "response_length": len(response),
            }

    # pylint: disable=unused-argument
    async def _aevaluate(
        self,
        response: str = "",
        **kwargs: Any,
    ) -> GraderScore:
        """
        Validate JSON format of the response.

        Attempts to parse the response string as JSON and returns a score indicating
        whether the parsing was successful. This grader does not compare against
        reference response since it only validates the format.

        Args:
            response (str, optional): The JSON string to validate. Defaults to empty string.
            **kwargs (Any): Additional keyword arguments (not used in this implementation).

        Returns:
            GraderScore: A score object containing:
                - score (float): 1.0 if the response is valid JSON, 0.0 otherwise
                - reason (str): Explanation of the validation result
                - metadata (dict): Detailed information about the validation including:
                    - is_valid (bool): Whether the JSON is valid
                    - response_length (int): Length of the response string
                    - error_message (str, optional): Error details if validation failed

        Example:
            >>> grader = JsonValidatorGrader()
            >>> result = await grader.aevaluate(
            ...     response='{"name": "Alice", "age": 30, "skills": ["Python", "AI"]}'
            ... )
            >>> print(result.score)  # 1.0 (valid JSON)
            >>>
            >>> invalid_result = await grader.aevaluate(
            ...     response='{"name": "Alice", "age": 30'  # Invalid JSON - missing closing brace
            ... )
            >>> print(result.score)  # 0.0 (invalid JSON)
        """
        is_valid, details = self._compute(response)

        if not is_valid:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details.get("error_message", "Invalid JSON"),
                metadata=details,
            )

        return GraderScore(
            name=self.name,
            score=1.0,
            reason="Valid JSON",
            metadata=details,
        )
