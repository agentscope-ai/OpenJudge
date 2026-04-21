# -*- coding: utf-8 -*-
"""
Trajectory Step Efficiency Grader

Evaluates the efficiency of the agent's trajectory — whether it achieves
the goal with a reasonable number of steps, avoiding unnecessary detours,
redundant tool calls, and wasteful exploration.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.agent.utils import (
    calculate_text_similarity,
    extract_action_observation_pairs,
)
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore


class TrajectoryStepEfficiencyGrader(BaseGrader):
    """
    Grader for evaluating the step efficiency of agent trajectories.

    This grader analyzes the agent's trajectory to measure:
    - Redundant steps (actions that repeat previous actions with no new information)
    - Detour steps (actions that don't contribute to goal progression)
    - Overall step economy (ratio of productive steps to total steps)

    A productive step is one that either provides new information (low similarity
    to previous observations) or represents a different action (not a duplicate).

    Attributes:
        redundancy_threshold: Similarity threshold for considering actions as redundant
        reference_step_count: Optional expected number of steps for the task

    Example:
        >>> import asyncio
        >>> grader = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.8)
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[...],
        ...     reference_step_count=5
        ... ))
        >>> print(f"Step efficiency score: {result.score}")
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.8,
        reference_step_count: Optional[int] = None,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize TrajectoryStepEfficiencyGrader.

        Args:
            redundancy_threshold: Threshold for considering observations as redundant (default: 0.8)
            reference_step_count: Optional expected number of steps. If provided, efficiency
                is also measured against this reference.
            strategy: Strategy for handling missing or invalid inputs
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            name="trajectory_step_efficiency",
            mode=GraderMode.POINTWISE,
            description="Evaluate the step efficiency of agent trajectories",
            strategy=strategy,
            **kwargs,
        )
        self.redundancy_threshold = redundancy_threshold
        self.reference_step_count = reference_step_count

    async def _aevaluate(
        self,
        messages: List[Dict[str, Any]],
        reference_step_count: Optional[int] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate trajectory step efficiency.

        Args:
            messages: List of message dicts containing agent interactions
            reference_step_count: Optional expected number of steps for the task.
                Overrides the instance attribute if provided.

        Returns:
            GraderScore: Step efficiency score with detailed metadata
        """
        ref_steps = reference_step_count or self.reference_step_count
        messages = [msg.get("message", msg) for msg in messages]
        action_obs_pairs = extract_action_observation_pairs(messages)

        if not action_obs_pairs:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No action-observation pairs found — unable to evaluate step efficiency",
                metadata={
                    "action_count": 0,
                    "productive_steps": 0,
                    "redundant_steps": 0,
                    "evaluable": False,
                },
            )

        total_steps = len(action_obs_pairs)
        redundant_steps = 0
        productive_steps = 0
        redundant_details = []

        # Track previous observations and action signatures for redundancy detection
        previous_observations = []
        previous_signatures = []

        for i, (action, observation) in enumerate(action_obs_pairs):
            # Check observation redundancy
            obs_redundant = False
            if previous_observations:
                max_sim = max(
                    calculate_text_similarity(observation, prev_obs)
                    for prev_obs in previous_observations
                )
                if max_sim >= self.redundancy_threshold:
                    obs_redundant = True

            # Check action redundancy
            import json

            function = action.get("function", {})
            current_signature = function.get("name", "")
            try:
                args = json.loads(function.get("arguments", "{}"))
                current_signature += ": " + ",".join(f"{k}={v}" for k, v in sorted(args.items()))
            except (json.JSONDecodeError, AttributeError):
                pass

            action_redundant = current_signature in previous_signatures

            if obs_redundant and action_redundant:
                redundant_steps += 1
                redundant_details.append({
                    "step": i,
                    "signature": current_signature,
                    "reason": "Both action and observation are redundant",
                })
            else:
                productive_steps += 1

            previous_observations.append(observation)
            previous_signatures.append(current_signature)

        # Calculate efficiency score: productive steps / total steps
        efficiency_ratio = productive_steps / total_steps if total_steps > 0 else 0.0

        # Calculate reference-based efficiency if reference is provided
        ref_efficiency = None
        if ref_steps and ref_steps > 0:
            if total_steps <= ref_steps:
                ref_efficiency = 1.0
            else:
                ref_efficiency = ref_steps / total_steps
            combined_score = 0.6 * efficiency_ratio + 0.4 * ref_efficiency
        else:
            combined_score = efficiency_ratio

        normalized_score = max(0.0, min(1.0, combined_score))

        # Build reason
        reason_parts = [
            f"Step efficiency: {productive_steps}/{total_steps} steps are productive "
            f"(ratio={efficiency_ratio:.3f})",
        ]
        if ref_steps:
            reason_parts.append(
                f"Reference-based efficiency: {ref_efficiency:.3f} "
                f"(reference={ref_steps} steps, actual={total_steps} steps)",
            )
        if redundant_steps > 0:
            reason_parts.append(f"Redundant steps: {redundant_steps}")

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason="; ".join(reason_parts),
            metadata={
                "total_steps": total_steps,
                "productive_steps": productive_steps,
                "redundant_steps": redundant_steps,
                "efficiency_ratio": efficiency_ratio,
                "reference_step_count": ref_steps,
                "reference_efficiency": ref_efficiency,
                "redundant_details": redundant_details,
                "redundancy_threshold": self.redundancy_threshold,
            },
        )
