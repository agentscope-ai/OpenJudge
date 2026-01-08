"""Code Execution Evaluation Module.

This module provides functionality for evaluating the correctness of generated code
by executing it against test cases. It includes the CodeExecutionGrader class which
tests code functionality using a testing framework that supports both call-based
and standard input code evaluation methods.

The module is designed to automatically extract code from responses, execute it
against predefined test cases, and provide scores based on the pass rate of those tests.
"""

import json
import re
import traceback
from typing import Any

from loguru import logger

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class CodeExecutionGrader(BaseGrader):
    """
    Executes code against test cases and evaluates correctness based on test case results.

    This reward model evaluates code by executing it against test cases using a testing framework
    that supports both call-based and standard input code evaluation methods.
    """

    def __init__(
        self,
        continuous: bool = True,
        timeout: int = 10,
        test_framework_available: bool = True,
        compute_score: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="code_execution",
            mode=GraderMode.POINTWISE,
            description="Executes code against test cases and evaluates correctness based on test case results",
            **kwargs,
        )

        self.continuous = continuous
        self.timeout = timeout
        self.test_framework_available = test_framework_available
        self.compute_score = compute_score

        try:
            from openjudge.graders.code._utils import compute_score

            self.compute_score = compute_score
            self.test_framework_available = True
        except ImportError:
            logger.warning(
                "Warning: Code testing framework not available. "
                "Please ensure openjudge.gallery.rm.code.prime_code is properly installed."
            )
            self.test_framework_available = False

        # Python code pattern in various formats
        self._python_code_pattern = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)
        # generic code formats
        self._generic_code_pattern = re.compile(r"```\n(.*?)\n```", flags=re.DOTALL)

    def _extract_code(self, content: str) -> str:
        """
        Extract code from content

        Args:
            content: Text content that may contain code blocks

        Returns:
            Extracted code
        """
        # Try to find Python code in various formats
        code_match = self._python_code_pattern.search(content)
        if code_match:
            return code_match.group(1)

        # Try other formats
        code_match = self._generic_code_pattern.search(content)
        if code_match:
            return code_match.group(1)

        # If no code block markers, assume the entire content is code
        return content

    async def aevaluate(self, response: str) -> GraderScore:
        """Evaluate code by executing it against test cases.

        Tests the functional correctness of generated code by executing it
        with predefined test cases. This grader checks if the code produces
        the expected outputs for given inputs, verifying its correctness.

        Args:
            response (str): The code to evaluate, potentially containing
                         markdown-style code blocks.

        Returns:
            GraderScore: The code execution evaluation result.
                - score (float): Execution score (0.0-1.0) based on test case results
                - reason (str): Explanation of the execution results
                - metadata (Dict[str, Any]): Additional information including:
                    * extracted_code: The code extracted from the response
                    * execution_results: Detailed results of code execution
                    * passed_tests: Number of passed test cases
                    * total_tests: Total number of test cases

        Example:
            >>> grader = CodeExecutionGrader()
            >>> code_response = "def add(a, b):\\n    return a + b"
            >>> result = await grader.aevaluate(code_response)
            >>> print(result.score, result.reason)
            1.0 Code executed successfully and passed all tests

            >>> # Example with failing code
            >>> bad_code = "def add(a, b):\\n    return a - b"
            >>> result = await grader.aevaluate(bad_code)
            >>> print(result.score, result.reason)
            0.0 Code execution failed or did not pass tests
        """
        # Extract code from response
        content = response
        extracted_code = self._extract_code(content)

        # Default values
        score = 0.0
        reason = "No evaluation performed"
        extra_data = {"extracted_code": extracted_code}

        # Check if testing framework is available
        if not self.test_framework_available:
            reason = "Code testing framework not available"
            extra_data["error"] = reason
        else:
            # Get test cases from sample metadata or label
            test_cases = None

            if not test_cases:
                reason = "No test cases available for evaluation"
            elif not extracted_code:
                score = 0.0
                reason = "No valid code extracted from response"
                extra_data["test_cases"] = test_cases
            else:
                # Convert test cases to string if needed
                if isinstance(test_cases, dict):
                    test_cases_str = json.dumps(test_cases)
                else:
                    test_cases_str = test_cases

                # Evaluate code using testing framework
                try:
                    success, _ = self.compute_score(
                        completion=extracted_code,
                        test_cases=test_cases_str,
                        continuous=self.continuous,
                    )

                    # Determine score based on success rate
                    if isinstance(success, bool):
                        pass_rate = 1.0 if success else 0.0
                    else:
                        pass_rate = float(success)

                    # Score is always between 0 and 1
                    score = pass_rate

                    # Generate reason based on results
                    if pass_rate == 1.0:
                        reason = "All test cases passed successfully"
                    elif pass_rate == 0.0:
                        reason = "No test cases passed"
                    else:
                        reason = f"Partial success: {pass_rate * 100:.1f}% of test cases passed"

                    # Include metadata in extra_data
                    extra_data = {
                        "extracted_code": extracted_code,
                        "test_cases": test_cases,
                        "pass_rate": pass_rate,
                    }

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    score = 0.0
                    reason = f"Evaluation error: {str(e)}"
                    extra_data = {
                        "extracted_code": extracted_code,
                        "test_cases": test_cases,
                        "error": str(e),
                        "traceback": error_traceback,
                    }

        # Single return statement at the end of the function
        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=extra_data,
        )
