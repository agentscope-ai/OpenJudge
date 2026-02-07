"""Patch Similarity Evaluator Module.

This module provides functionality to calculate similarity between response
code patches and reference (oracle) patches using difflib.SequenceMatcher.
It includes the PatchSimilarityGrader class which evaluates the quality of
code modifications by comparing them with reference implementations.

The module is designed for use in code generation evaluation scenarios where
the similarity between a response patch and a correct reference patch needs
to be quantified.
"""

import difflib

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class PatchSimilarityGrader(BaseGrader):
    """
    Calculate similarity between response patch and oracle patch using difflib.SequenceMatcher.

    This reward measures how similar the response patch is to the reference patch,
    providing a similarity score and detailed diff information.
    """

    def __init__(self, strategy: BaseEvaluationStrategy | None = None):
        """
        Initialize PatchSimilarityGrader.
        Args:
            strategy (BaseEvaluationStrategy): The strategy to use for grading.
        """
        super().__init__(
            name="patch_similarity",
            mode=GraderMode.POINTWISE,
            description="Calculate similarity between response patch and oracle patch using difflib.SequenceMatcher",
            strategy=strategy,
        )

    async def _aevaluate(self, response: str, reference_response: str) -> GraderScore:
        """Calculate similarity between response and reference patches.

        Uses difflib.SequenceMatcher to calculate the similarity ratio between
        a response patch and a reference (oracle) patch. This is useful for
        evaluating the quality of code modifications or response diffs.

        Args:
            response (str): The response patch or code modification to evaluate.
            reference_response (str): The reference (oracle) patch that is considered correct.

        Returns:
            GraderScore: The patch similarity evaluation result.
                - score (float): Similarity score between response and reference patches (0.0-1.0)
                - reason (str): Explanation of the similarity score
                - metadata (Dict[str, Any]): Additional information including:
                    * similarity: The calculated similarity ratio
                    * response: The response patch
                    * reference_response: The reference patch
                    * opcodes: Detailed diff operations from SequenceMatcher

        Example:
            >>> grader = PatchSimilarityGrader()
            >>> response = "def add(a, b):\\n    return a + b"
            >>> reference_response = "def add(x, y):\\n    return x + y"
            >>> result = await grader.aevaluate(response, reference_response)
            >>> print(result.score, result.reason)
            0.8 Patch similarity: 0.800 based on sequence matching
        """

        # Use SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, response, reference_response)
        similarity = matcher.ratio()

        # Get detailed diff information
        opcodes = list(matcher.get_opcodes())

        return GraderScore(
            name=self.name,
            score=similarity,
            reason=f"Patch similarity: {similarity:.3f} based on sequence matching",
            metadata={
                "similarity": similarity,
                "response": response,
                "reference_response": reference_response,
                "opcodes": opcodes,
            },
        )
