"""Length Penalty Grader Module

This module provides functionality to apply penalties to text based on its length.
It contains the LengthPenaltyGrader class which applies penalties for content
that is either too short or too long according to configured thresholds.
"""

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class LengthPenaltyGrader(BaseGrader):
    """
    Text length based penalty for content that is too short or too long.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        penalty_rate: float = 0.01,
    ):
        """
        Initialize the LengthPenaltyGrader.
        Args:
            min_length: Minimum length of the content
            max_length: Maximum length of the content
            penalty_rate: Penalty rate for each character beyond the maximum length
        """
        super().__init__(
            name="length_penalty",
            grader_mode="content",
            mode=GraderMode.POINTWISE,
            description="Text length based penalty for content that is too short or too long.",
        )

        self.min_length = min_length
        self.max_length = max_length
        self.penalty_rate = penalty_rate

    async def aevaluate(self, response: str) -> GraderScore:
        """
        Calculate length-based penalty for text content.

        This method evaluates the length of the provided text and applies penalties
        if the text is too short or too long according to configured thresholds.

        Penalty calculation:
        - If length < min_length: penalty = -(min_length - length) * penalty_rate
        - If length > max_length: penalty = -(length - max_length) * penalty_rate
        - Otherwise: penalty = 0.0

        Args:
            response: The text content to evaluate for length.

        Returns:
            GraderScore: A GraderScore object containing:
                - score: The calculated penalty (negative value or 0.0)
                - reason: Explanation of why the penalty was applied or not
                - metadata: Dictionary with detailed information:
                    * length: Actual length of the text
                    * min_length: Configured minimum length
                    * max_length: Configured maximum length
                    * penalty: The calculated penalty value

        Examples:
            >>> grader = LengthPenaltyGrader(min_length=5, max_length=20, penalty_rate=0.1)
            >>> result = await grader.aevaluate("This is a good length")
            >>> print(result.score)
            0.0

            >>> result = await grader.aevaluate("Too short")
            >>> print(result.score < 0)
            True

            >>> result = await grader.aevaluate("This text is way too long to be acceptable for this particular grader")
            >>> print(result.score < 0)
            True
        """

        length = len(response)

        penalty = 0.0
        reason_parts = []

        if length < self.min_length:
            penalty = -(self.min_length - length) * self.penalty_rate
            reason_parts.append(f"Too short: {length} < {self.min_length}")
        elif length > self.max_length:
            penalty = -(length - self.max_length) * self.penalty_rate
            reason_parts.append(f"Too long: {length} > {self.max_length}")
        else:
            reason_parts.append(
                f"Length acceptable: {self.min_length} <= {length} <= {self.max_length}",
            )

        return GraderScore(
            name=self.name,
            score=penalty,
            reason="; ".join(reason_parts),
            metadata={
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "penalty": penalty,
            },
        )
