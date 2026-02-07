# -*- coding: utf-8 -*-
"""
Observation Information Gain Grader
This module provides a grader for evaluating information gain and redundancy
in observation observations.
"""
import math
from typing import Any, Dict, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.agent.utils import (
    calculate_text_similarity,
    extract_action_observation_pairs,
)
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore


class ObservationInformationGainGrader(BaseGrader):
    """
    Grader for information gain and redundancy in observation observations.
    For each observation, compares it to all previous observations:
    - If the maximum similarity is low, reward for information gain
    - If the maximum similarity is high, penalize for redundancy
    Attributes:
        similarity_threshold: Threshold for considering observations as redundant
    Example:
        >>> import asyncio
        >>> grader = ObservationInformationGainGrader(similarity_threshold=0.5)
        >>> result = asyncio.run( grader.aevaluate(
        ...     messages=[...],  # List of message dicts
        ... ))
        >>> print(f"Info gain score: {result.score}")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the ObservationInformationGainGrader.

        Args:
            similarity_threshold: Threshold for considering observations as redundant
            strategy: Strategy for handling missing or invalid inputs
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            name="observation_information_gain",
            mode=GraderMode.POINTWISE,
            description="Evaluate information gain and redundancy in observation observations",
            strategy=strategy,
            **kwargs,
        )
        self.similarity_threshold = similarity_threshold

    async def _aevaluate(
        self,
        messages: List[Dict[str, Any]],
    ) -> GraderScore:
        """
        Evaluate information gain and redundancy in observation observations.
        Args:
            messages: List of message dicts containing agent interactions
        Returns:
            GraderScore: Information gain score with details
        """
        messages = [msg.get("message", msg) for msg in messages]
        action_obs_pairs = extract_action_observation_pairs(messages)
        if not action_obs_pairs:
            reason = "No action-observation pairs found - unable to evaluate information gain"
            return GraderScore(
                name=self.name,
                score=0.0,  # Neutral score: cannot evaluate without actions
                reason=reason,
                metadata={
                    "action_count": 0,
                    "reason": reason,
                    "evaluable": False,
                },
            )
        rewards = []
        previous_observations = []
        similarity_list = []
        # pylint: disable=unused-variable
        for action, observation in action_obs_pairs:
            # Skip if observation is empty or too short
            if not observation or len(observation.strip()) < 10:
                continue
            # Calculate max similarity to all previous observations
            if previous_observations:
                similarities = [calculate_text_similarity(observation, prev_obs) for prev_obs in previous_observations]
                max_similarity = max(similarities)
            else:
                max_similarity = 0.0
            similarity_list.append(max_similarity)
            # Info gain score: 1 - max_similarity
            info_score = 1.0 - max_similarity
            # Reward logic with exponential decay:
            # Apply exponential penalty based on how much similarity exceeds threshold
            # When max_similarity < threshold: penalty_factor ≈ 1.0 (no penalty)
            # When max_similarity >= threshold: penalty_factor decreases exponentially
            similarity_excess = max(0, max_similarity - self.similarity_threshold)
            penalty_factor = math.exp(-2 * similarity_excess)
            reward = info_score * penalty_factor
            rewards.append(reward)
            previous_observations.append(observation)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        # With exponential decay, rewards are naturally in [0, 1] range
        # No negative values, so direct normalization is sufficient
        normalized_score = max(0.0, min(1.0, avg_reward))
        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=f"Average info gain score across {len(rewards)} observation steps: {avg_reward:.3f}",
            metadata={
                "raw_average": avg_reward,
                "observation_count": len(action_obs_pairs),
                "each_turn_rewards": rewards,
                "each_turn_similarity": similarity_list,
                "similarity_threshold": self.similarity_threshold,
            },
        )
