"""
JSON Matching and Validation Metrics

JSON format validation and deep comparison metrics.
Restructured to work with Grader framework.
"""

import json
from typing import Any

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class JsonMatchGrader(Grader):
    """
    JSON Deep Match Grader

    Recursively compares JSON structures element by element.

    Attributes:
        name: Grader name
        strict_order: Whether to strictly compare list order
        ignore_extra_keys: Whether to ignore extra keys in candidate dict

    Example:
        >>> grader = JsonMatchGrader()
        >>> result = await grader.evaluate(
        ...     reference='{"name": "Alice", "age": 30}',
        ...     candidate='{"name": "Alice", "age": 30}'
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
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.strict_order = strict_order
        self.ignore_extra_keys = ignore_extra_keys

    def _json_match(self, sampled: Any, correct: Any) -> bool:
        """Recursively compare JSON structures"""
        if sampled is None or correct is None:
            return sampled == correct

        if isinstance(sampled, dict):
            if not isinstance(correct, dict):
                return False

            if self.ignore_extra_keys:
                return all(
                    self._json_match(sampled.get(key), correct.get(key))
                    for key in correct.keys()
                )
            else:
                all_keys = set(sampled.keys()) | set(correct.keys())
                return all(
                    self._json_match(sampled.get(key), correct.get(key))
                    for key in all_keys
                )

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

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """
        Compute JSON match

        Returns:
            tuple[bool, dict]: (matched, details)
        """
        # Parse candidate JSON
        try:
            candidate_json = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as e:
            return False, {
                "matched": False,
                "error": "candidate_parse_error",
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

        matched = self._json_match(candidate_json, reference_json)

        details = {
            "matched": matched,
            "strict_order": self.strict_order,
            "ignore_extra_keys": self.ignore_extra_keys,
        }

        return matched, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate JSON match"""
        matched, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(
                score=0.0, reason=details["error_message"], metadata=details
            )

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"JSON match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )


class JsonValidatorGrader(Grader):
    """
    JSON Format Validator Grader

    Validates if the candidate text is valid JSON.

    Attributes:
        name: Grader name

    Example:
        >>> grader = JsonValidatorGrader()
        >>> result = await grader.evaluate(
        ...     reference="",  # reference not needed
        ...     candidate='{"valid": "json"}'
        ... )
        >>> print(f"Valid: {result.score}")
    """

    def __init__(
        self,
        name: str = "json_validator",
        description: str = "JSON format validation metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )

    def _compute(self, candidate: str) -> tuple[bool, dict]:
        """
        Validate JSON format

        Returns:
            tuple[bool, dict]: (is_valid, details)
        """
        try:
            json.loads(candidate)
            return True, {"is_valid": True, "candidate_length": len(candidate)}
        except json.JSONDecodeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"JSON decode error: {str(e)}",
                "candidate_length": len(candidate),
            }
        except TypeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"Type error: {str(e)}",
                "candidate_length": len(candidate),
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "error_message": f"Unexpected error: {str(e)}",
                "candidate_length": len(candidate),
            }

    async def evaluate(
        self, reference: str = "", candidate: str = "", **kwargs
    ) -> GraderScore:
        """Validate JSON format"""
        is_valid, details = self._compute(candidate)

        if not is_valid:
            return GraderScore(
                score=0.0,
                reason=details.get("error_message", "Invalid JSON"),
                metadata=details,
            )

        return GraderScore(
            score=1.0,
            reason="Valid JSON",
            metadata=details,
        )


__all__ = [
    "JsonMatchGrader",
    "JsonValidatorGrader",
]
