"""
Reasoning Format Grader Module.

This module provides functionality to evaluate if response content follows the required
thinking and answer format with proper XML-style tags. It contains the ReasoningFormatGrader
class which checks for the presence of thinking and answer tags in the response text.
"""

import re
from typing import Any

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class ReasoningFormatGrader(BaseGrader):
    """
    Check format reward for thinking format and answer format with proper tags.

    This reward verifies if the response content follows the required format
    with proper <think> and <answer> tags.
    """

    def __init__(
        self, think_token: str = "think", answer_token: str = "answer", strategy: BaseEvaluationStrategy | None = None
    ):
        """
        Initialize the ReasoningFormatGrader.
        Args:
            think_token: The token used for thinking tags. Defaults to "think".
            answer_token: The token used for answer tags. Defaults to "answer".
            strategy: The strategy to use for evaluation. Defaults to None.
        """
        super().__init__(
            name="format_reward",
            mode=GraderMode.POINTWISE,
            description="Check format reward for thinking format and answer format with proper tags.",
            strategy=strategy,
        )
        self.think_token = think_token
        self.think_pattern = re.compile(f"<{self.think_token}>.*?</{self.think_token}>", flags=re.DOTALL)

        self.answer_token = answer_token
        self.answer_pattern = re.compile(f"<{self.answer_token}>.*?</{self.answer_token}>", flags=re.DOTALL)

    # pylint: disable=unused-argument
    async def _aevaluate(self, response: str, *args: Any, **kwargs: Any) -> GraderScore:
        """
        Check format and calculate reward for reasoning tags.

        This method evaluates if the given answer follows the required format with proper
        thinking and answer tags. It checks for the presence of both thinking and answer
        tags and assigns a score of 1.0 only if both are present, otherwise 0.0.

        Args:
            response: The response text to evaluate for proper formatting.
            *args: Additional positional arguments (not used in this implementation).
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: 1.0 if both thinking and answer tags are present, 0.0 otherwise
                - reason: Explanation of the evaluation result
                - metadata: Dictionary with detailed information:
                    * has_think_tag: Whether thinking tags are present
                    * has_answer_tag: Whether answer tags are present
                    * total_reward: The calculated reward score
                    * think_token: The token used for thinking tags
                    * answer_token: The token used for answer tags

        Examples:
            >>> grader = ReasoningFormatGrader()
            >>> result = await grader.aevaluate("Some text without tags")
            >>> print(result.score)
            0.0

            >>> result = await grader.aevaluate("<think>Thought process</think>\\n<answer>Final answer</answer>")
            >>> print(result.score)
            1.0
        """
        # Check thinking format tags
        has_think_tag = bool(self.think_pattern.search(response))

        # Check answer format tags
        has_answer_tag = bool(self.answer_pattern.search(response))

        # Calculate reward
        reward = 1.0 if has_think_tag and has_answer_tag else 0.0
        reasons = []

        if not has_think_tag:
            reasons.append(
                f"Missing <{self.think_token}></{self.think_token}> tags",
            )

        if not has_answer_tag:
            reasons.append(
                f"Missing <{self.answer_token}></{self.answer_token}> tags",
            )

        if reward == 1.0:
            reasons.append("All format requirements met")

        return GraderScore(
            name=self.name,
            score=reward,
            reason="; ".join(reasons),
            metadata={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "total_reward": reward,
                "think_token": self.think_token,
                "answer_token": self.answer_token,
            },
        )
