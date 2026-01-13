# -*- coding: utf-8 -*-
"""
JSON Match Grader Module

This module provides functionality to perform deep comparison of JSON structures.
It contains the JsonMatchGrader class which recursively compares JSON elements
and calculates similarity scores based on structural matching.
"""

import json
from typing import Any

from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore


class JsonMatchGrader(BaseGrader):
    """
    JSON Deep Match Grader

    Recursively compares JSON structures element by element.

    Attributes:
        name: Grader name
        strict_order: Whether to strictly compare list order
        ignore_extra_keys: Whether to ignore extra keys in response dict

    Example:
        >>> grader = JsonMatchGrader()
        >>> result = await grader.aevaluate(
        ...     reference='{"name": "Alice", "age": 30}',
        ...     response='{"name": "Alice", "age": 30}'
        ... )
        >>> print(f"Matched: {result.score}")
    """

    def __init__(
        self,
        name: str = "json_match",
        strict_order: bool = True,
        ignore_extra_keys: bool = False,
        description: str = "JSON deep comparison metric",
    ):
        super().__init__(
            name=name,
            mode=GraderMode.POINTWISE,
            description=description,
        )
        self.strict_order = strict_order
        self.ignore_extra_keys = ignore_extra_keys

    # pylint: disable=too-many-return-statements
    def _json_match(self, sampled: Any, correct: Any) -> bool:
        """Recursively compare JSON structures"""
        if sampled is None or correct is None:
            return sampled == correct

        if isinstance(sampled, dict):
            if not isinstance(correct, dict):
                return False

            if self.ignore_extra_keys:
                return all(self._json_match(sampled.get(key), correct.get(key)) for key in correct.keys())
            else:
                all_keys = set(sampled.keys()) | set(correct.keys())
                return all(self._json_match(sampled.get(key), correct.get(key)) for key in all_keys)

        elif isinstance(sampled, list):
            if not isinstance(correct, list):
                return False

            if len(sampled) != len(correct):
                return False

            if self.strict_order:
                return all(self._json_match(s, c) for s, c in zip(sampled, correct))
            else:
                if len(sampled) == 0:
                    return True

                used = [False] * len(correct)
                for s_item in sampled:
                    found_match = False
                    for i, c_item in enumerate(correct):
                        if not used[i] and self._json_match(s_item, c_item):
                            used[i] = True
                            found_match = True
                            break
                    if not found_match:
                        return False
                return True

        return sampled == correct

    def _compute(self, reference: str, response: str) -> tuple[bool, dict]:
        """
        Compute JSON match

        Returns:
            tuple[bool, dict]: (matched, details)
        """
        # Parse response JSON
        try:
            response_json = json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            return False, {
                "matched": False,
                "error": "response_parse_error",
                "error_message": str(e),
            }

        # Parse reference JSON
        try:
            reference_json = json.loads(reference)
        except (json.JSONDecodeError, TypeError) as e:
            return False, {
                "matched": False,
                "error": "reference_parse_error",
                "error_message": str(e),
            }

        matched = self._json_match(response_json, reference_json)

        details = {
            "matched": matched,
            "strict_order": self.strict_order,
            "ignore_extra_keys": self.ignore_extra_keys,
        }

        return matched, details

    async def aevaluate(
        self,
        reference_response: str,
        response: str,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate JSON match between reference response and response.

        Performs a deep comparison of two JSON strings to determine if they match
        according to configured matching rules (strict order, ignoring extra keys, etc).

        Args:
            reference_response (str): The reference JSON string that serves as the expected result
            response (str): The actual JSON string to be evaluated
            **kwargs (Any): Additional keyword arguments (not used in this implementation)

        Returns:
            GraderScore: A score object containing:
                - score (float): 1.0 if JSON structures match, 0.0 otherwise
                - reason (str): Explanation of the matching result
                - metadata (dict): Detailed information about the comparison including:
                    - matched (bool): Whether the JSON structures matched
                    - strict_order (bool): Whether list order was strictly enforced
                    - ignore_extra_keys (bool): Whether extra keys in response were ignored
                    - error information if parsing failed

        Example:
            >>> grader = JsonMatchGrader(strict_order=False)
            >>> result = await grader.aevaluate(
            ...     reference_response='{"name": "Alice", "hobbies": ["reading", "swimming"]}',
            ...     response='{"name": "Alice", "hobbies": ["swimming", "reading"]}'
            ... )
            >>> print(result.score)  # 1.0 (matches because strict_order=False)
        """
        matched, details = self._compute(reference_response, response)

        if "error" in details:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details["error_message"],
                metadata=details,
            )

        return GraderScore(
            name=self.name,
            score=1.0 if matched else 0.0,
            reason=f"JSON match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )
