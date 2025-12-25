"""Code Style Evaluation Module.

This module provides functionality for evaluating the style of Python code,
including indentation consistency and naming conventions. It includes the
CodeStyleGrader class which performs basic style checks on code blocks
extracted from responses.

The module is designed to analyze Python code for adherence to common style
guidelines such as consistent indentation (spaces vs tabs) and proper naming
conventions (snake_case for functions and variables).
"""

import re
from typing import Any, Dict

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class CodeStyleGrader(BaseGrader):
    """Basic code style checking including indentation consistency and naming conventions."""

    def __init__(self):
        super().__init__(
            name="code_style",
            mode=GraderMode.POINTWISE,
            description="Basic code style checking including indentation consistency and naming conventions.",
        )

    def _check_indentation(self, code: str) -> tuple[bool, str]:
        """Check indentation consistency"""
        lines = code.split("\n")
        indent_type = None  # 'spaces' or 'tabs'

        for line in lines:
            if line.strip():  # Non-empty line
                leading = len(line) - len(line.lstrip())
                if leading > 0:
                    if line.startswith(" "):
                        if indent_type is None:
                            indent_type = "spaces"
                        elif indent_type != "spaces":
                            return (
                                False,
                                "Mixed indentation types (spaces and tabs)",
                            )
                    elif line.startswith("\t"):
                        if indent_type is None:
                            indent_type = "tabs"
                        elif indent_type != "tabs":
                            return (
                                False,
                                "Mixed indentation types (spaces and tabs)",
                            )

        return True, "Consistent indentation"

    def _check_naming(self, code: str) -> tuple[float, str]:
        """Check naming conventions"""
        # Simple naming check
        function_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        variable_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*="

        functions = re.findall(function_pattern, code)
        variables = re.findall(variable_pattern, code)

        total_names = len(functions) + len(variables)
        if total_names == 0:
            return 1.0, "No names to check"

        good_names = 0

        # Check function names (should be snake_case)
        for func in functions:
            if re.match(r"^[a-z_][a-z0-9_]*$", func):
                good_names += 1

        # Check variable names (should be snake_case)
        for var in variables:
            if re.match(r"^[a-z_][a-z0-9_]*$", var):
                good_names += 1

        score = good_names / total_names
        return (
            score,
            f"Naming convention: {good_names}/{total_names} names follow snake_case",
        )

    async def aevaluate(self, response: str) -> GraderScore:
        """Evaluate code style in the provided response.

        Performs basic code style checking including indentation consistency and
        naming conventions. Extracts Python code blocks from markdown-style code
        fences and evaluates each one for style compliance.

        Args:
            response (str): The response containing code blocks to check for style issues.
                        Code blocks should be enclosed in markdown-style fences (```python).

        Returns:
            GraderScore: The code style evaluation result.
                - score (float): Average style score across all code blocks (0.0-1.0)
                - reason (str): Explanation of the style check result with details
                - metadata (Dict[str, Any]): Additional information including:
                    * code_blocks: List of extracted code blocks
                    * average_score: Average style score
                    * details: Detailed breakdown of style checks

        Example:
            >>> grader = CodeStyleGrader()
            >>> response = "Here's a function with correct style:"
            >>>          "```python\\ndef calculate_sum(a,b):\\n    return a+b\\n```"
            >>> result = await grader.aevaluate(response=response)
            >>> print(result.score, result.reason)
            1.0 Code style score: 1.000; Consistent indentation; Naming convention: 2/2 names follow snake_case

            >>> # Example with style issues
            >>> response = "Here's a function with style issues:\\n```python\\ndef CalculateSum(a,b):\\n
                return a+b\\n```"
            >>> result = await grader.aevaluate(response=response)
            >>> print(result.score, result.reason)
            0.5 Code style score: 0.500; Consistent indentation; Naming convention: 1/2 names follow snake_case
        """
        # Extract code blocks
        code_pattern = r"```(?:python)?\s*\n(.*?)\n\s*```"
        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if not code_blocks:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No code blocks found to check style",
                metadata={"code_blocks": []},
            )

        total_score = 0.0
        details = []

        for i, code in enumerate(code_blocks):
            block_score = 0.0

            # Check indentation
            indent_ok, indent_msg = self._check_indentation(code)
            if indent_ok:
                block_score += 0.5
            details.append(f"Block {i}: {indent_msg}")

            # Check naming
            naming_score, naming_msg = self._check_naming(code)
            block_score += naming_score * 0.5
            details.append(f"Block {i}: {naming_msg}")

            total_score += block_score

        # Average score
        average_score = total_score / len(code_blocks)
        return GraderScore(
            name=self.name,
            score=average_score,
            reason=f"Code style score: {average_score:.3f}; " + "; ".join(details),
            metadata={
                "code_blocks": code_blocks,
                "average_score": average_score,
                "details": details,
            },
        )

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return the docstring of the aevaluate method."""
        return {"aevaluate": CodeStyleGrader.aevaluate.__doc__}
