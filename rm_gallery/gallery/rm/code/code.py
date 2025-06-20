import ast
import difflib
import json
import re
import traceback

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult


@RewardRegistry.register("code_syntax_check")
class SyntaxCheckReward(BasePointWiseReward):
    """
    Check code syntax using Abstract Syntax Tree
    """

    name: str = Field(default="syntax_check", description="Syntax check reward")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check code syntax

        Args:
            sample: Data sample containing code content

        Returns:
            RewardResult: Reward result containing syntax check results
        """
        content = sample.output[0].answer.content

        # Extract code blocks
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        code_blocks = re.findall(code_pattern, content, re.DOTALL)

        if not code_blocks:
            # No code blocks, return neutral score
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason="No code blocks found to check",
                    )
                ],
                extra_data={"code_blocks": [], "syntax_errors": []},
            )

        syntax_errors = []
        valid_blocks = 0

        for i, code in enumerate(code_blocks):
            try:
                ast.parse(code.strip())
                valid_blocks += 1
            except SyntaxError as e:
                syntax_errors.append(
                    {"block": i, "error": str(e), "line": e.lineno, "offset": e.offset}
                )

        # Calculate score: ratio of valid code blocks
        score = valid_blocks / len(code_blocks) if code_blocks else 0.0

        # Apply penalty if syntax errors exist
        if syntax_errors:
            score -= 0.5

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=f"Syntax check: {valid_blocks}/{len(code_blocks)} blocks valid, {len(syntax_errors)} errors",
                )
            ],
            extra_data={
                "code_blocks": code_blocks,
                "valid_blocks": valid_blocks,
                "total_blocks": len(code_blocks),
                "syntax_errors": syntax_errors,
            },
        )


@RewardRegistry.register("code_style")
class CodeStyleReward(BasePointWiseReward):
    """
    Basic code style checking
    """

    name: str = Field(default="code_style", description="Code style reward")

    def _check_indentation(self, code: str) -> tuple[bool, str]:
        """Check indentation consistency"""
        lines = code.split("\n")
        indent_type = None  # 'spaces' or 'tabs'
        indent_size = None

        for line in lines:
            if line.strip():  # Non-empty line
                leading = len(line) - len(line.lstrip())
                if leading > 0:
                    if line.startswith(" "):
                        if indent_type is None:
                            indent_type = "spaces"
                            indent_size = leading
                        elif indent_type != "spaces":
                            return False, "Mixed indentation types (spaces and tabs)"
                    elif line.startswith("\t"):
                        if indent_type is None:
                            indent_type = "tabs"
                        elif indent_type != "tabs":
                            return False, "Mixed indentation types (spaces and tabs)"

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

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check code style

        Args:
            sample: Data sample containing code

        Returns:
            RewardResult: Reward result containing code style score
        """
        content = sample.output[0].answer.content

        # Extract code blocks
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        code_blocks = re.findall(code_pattern, content, re.DOTALL)

        if not code_blocks:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason="No code blocks found to check style",
                    )
                ],
                extra_data={"code_blocks": []},
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

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=average_score,
                    reason=f"Code style score: {average_score:.3f}; "
                    + "; ".join(details),
                )
            ],
            extra_data={
                "average_score": average_score,
                "code_blocks_count": len(code_blocks),
                "details": details,
            },
        )


@RewardRegistry.register("code_patch_similarity")
class PatchSimilarityReward(BasePointWiseReward):
    """
    Calculate similarity between generated patch and oracle patch using difflib.SequenceMatcher.

    This reward measures how similar the generated patch is to the reference patch,
    providing a similarity score and detailed diff information.
    """

    name: str = Field(default="patch_similarity", description="Patch similarity reward")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Calculate patch similarity.

        Args:
            sample: Data sample containing generated patch

        Returns:
            RewardResult: Reward result containing similarity score
        """
        generated = sample.output[0].answer.content.strip()
        reference = sample.output[0].answer.label.get("reference", "").strip()

        # Use SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, generated, reference)
        similarity = matcher.ratio()

        # Get detailed diff information
        opcodes = list(matcher.get_opcodes())

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=similarity,
                    reason=f"Patch similarity: {similarity:.3f} based on sequence matching",
                )
            ],
            extra_data={
                "similarity": similarity,
                "generated": generated,
                "reference": reference,
                "opcodes": opcodes,
            },
        )


class CodeExecutionReward(BasePointWiseReward):
    """
    Evaluate code by executing it against test cases

    This reward model evaluates code by executing it against test cases using a testing framework
    that supports both call-based and standard input code evaluation methods.
    """

    name: str = Field(default="code_execution", description="Code execution reward")
    continuous: bool = Field(
        default=True, description="Use continuous scoring (partial credit)"
    )
    timeout: int = Field(
        default=10, description="Timeout in seconds for code execution"
    )

    def __init__(self, **data):
        super().__init__(**data)
        try:
            from rm_gallery.gallery.rm.code.prime_code import compute_score

            self.compute_score = compute_score
            self.test_framework_available = True
        except ImportError:
            print(
                "Warning: Code testing framework not available. Please ensure rm_gallery.gallery.rm.code.prime_code is properly installed."
            )
            self.test_framework_available = False

    def _extract_code(self, content: str) -> str:
        """
        Extract code from content

        Args:
            content: Text content that may contain code blocks

        Returns:
            Extracted code
        """
        # Try to find Python code in various formats
        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try other formats
        code_match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # If no code block markers, assume the entire content is code
        return content

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Evaluate code against test cases

        Args:
            sample: Data sample containing code content and test cases

        Returns:
            RewardResult: Reward result containing evaluation score
        """
        # Extract code from response
        content = sample.output[0].answer.content
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
            if sample.metadata and "inputs_outputs" in sample.metadata:
                test_cases = sample.metadata["inputs_outputs"]
            elif (
                sample.output[0].answer.label
                and "inputs_outputs" in sample.output[0].answer.label
            ):
                test_cases = sample.output[0].answer.label["inputs_outputs"]

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
                    success, metadata = self.compute_score(
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
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=reason,
                )
            ],
            extra_data=extra_data,
        )
