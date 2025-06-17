import ast
import difflib
import re

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult


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
