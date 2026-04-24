# -*- coding: utf-8 -*-
"""
Tool Usage Efficiency Grader

Evaluates whether the agent uses tools efficiently — combining related queries,
avoiding redundant calls, and selecting tools that maximize information gain per call.
"""

import json
from collections import Counter
from typing import Any, Dict, List

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.agent.utils import (
    calculate_text_similarity,
    extract_action_observation_pairs,
)
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore


class ToolUsageEfficiencyGrader(BaseGrader):
    """
    Grader for evaluating tool usage efficiency in agent trajectories.

    This grader analyzes how efficiently the agent uses its tools by measuring:
    - Redundant tool calls (same tool called with identical or very similar arguments)
    - Information gain per tool call (how much new information each call yields)
    - Tool diversity (whether the agent leverages multiple available tools effectively)

    Attributes:
        redundancy_threshold: Similarity threshold for considering tool calls as redundant
        information_gain_threshold: Threshold for considering observations as providing
            new information (default: 0.3, lower = more similar = less new info)

    Example:
        >>> import asyncio
        >>> grader = ToolUsageEfficiencyGrader(redundancy_threshold=0.8)
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[...],
        ... ))
        >>> print(f"Tool usage efficiency score: {result.score}")
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.8,
        information_gain_threshold: float = 0.3,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize ToolUsageEfficiencyGrader.

        Args:
            redundancy_threshold: Threshold for considering tool calls as redundant (default: 0.8)
            information_gain_threshold: Threshold below which observations are considered
                low information gain (default: 0.3, meaning obs similarity > 0.7 = low gain)
            strategy: Strategy for handling missing or invalid inputs
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            name="tool_usage_efficiency",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool usage efficiency in agent trajectories",
            strategy=strategy,
            **kwargs,
        )
        self.redundancy_threshold = redundancy_threshold
        self.information_gain_threshold = information_gain_threshold

    def _extract_tool_signature(self, action: Dict[str, Any]) -> str:
        """Extract a normalized tool call signature for comparison."""
        function = action.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            sorted_args = sorted(args.items())
            args_str = ",".join(f"{k}={v}" for k, v in sorted_args)
            return f"{name}({args_str})"
        except (json.JSONDecodeError, AttributeError):
            return f"{name}({raw_args})"

    async def _aevaluate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate tool usage efficiency.

        Args:
            messages: List of message dicts containing agent interactions

        Returns:
            GraderScore: Tool usage efficiency score with detailed metadata
        """
        messages = [msg.get("message", msg) for msg in messages]
        action_obs_pairs = extract_action_observation_pairs(messages)

        if not action_obs_pairs:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No action-observation pairs found — unable to evaluate tool usage efficiency",
                metadata={
                    "tool_call_count": 0,
                    "evaluable": False,
                },
            )

        total_calls = len(action_obs_pairs)

        # 1. Detect redundant tool calls (same signature repeated)
        signatures = [self._extract_tool_signature(action) for action, _ in action_obs_pairs]
        signature_counts = Counter(signatures)
        duplicate_calls = sum(count - 1 for count in signature_counts.values() if count > 1)
        unique_calls = len(signature_counts)

        # 2. Calculate information gain per call
        previous_observations = []
        low_gain_calls = 0
        info_gains = []

        for action, observation in action_obs_pairs:
            if previous_observations:
                max_sim = max(calculate_text_similarity(observation, prev_obs) for prev_obs in previous_observations)
                info_gain = 1.0 - max_sim
            else:
                info_gain = 1.0  # First call always provides new info

            info_gains.append(info_gain)
            if info_gain < self.information_gain_threshold:
                low_gain_calls += 1
            previous_observations.append(observation)

        avg_info_gain = sum(info_gains) / len(info_gains) if info_gains else 0.0

        # 3. Calculate tool diversity
        tool_names = []
        for action, _ in action_obs_pairs:
            function = action.get("function", {})
            tool_names.append(function.get("name", "unknown"))
        unique_tools = len(set(tool_names))
        total_tools = len(tool_names)
        diversity_ratio = unique_tools / total_tools if total_tools > 0 else 0.0

        # 4. Compute overall efficiency score
        # Redundancy penalty: duplicate calls reduce efficiency
        redundancy_penalty = duplicate_calls / total_calls if total_calls > 0 else 0.0
        # Information gain score: average info gain across calls
        info_gain_score = avg_info_gain
        # Diversity score: using different tools is generally more efficient
        # But we don't want to over-penalize single-tool scenarios
        diversity_score = min(1.0, diversity_ratio + 0.3)  # Floor at 0.3 for single tool

        # Weighted combination
        efficiency_score = 0.4 * info_gain_score + 0.4 * (1.0 - redundancy_penalty) + 0.2 * diversity_score

        normalized_score = max(0.0, min(1.0, efficiency_score))

        # Build reason
        reason = (
            f"Tool usage efficiency: {normalized_score:.3f} "
            f"(info_gain={avg_info_gain:.3f}, "
            f"redundancy={duplicate_calls}/{total_calls}, "
            f"diversity={unique_tools}/{total_tools})"
        )

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata={
                "total_tool_calls": total_calls,
                "unique_tool_calls": unique_calls,
                "duplicate_calls": duplicate_calls,
                "redundancy_penalty": redundancy_penalty,
                "avg_info_gain": avg_info_gain,
                "low_gain_calls": low_gain_calls,
                "info_gains": info_gains,
                "unique_tools": unique_tools,
                "diversity_ratio": diversity_ratio,
                "redundancy_threshold": self.redundancy_threshold,
                "information_gain_threshold": self.information_gain_threshold,
            },
        )
