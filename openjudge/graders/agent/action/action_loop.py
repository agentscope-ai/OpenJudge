# -*- coding: utf-8 -*-
"""
Action Loop Detection Grader
This module provides a grader for detecting and penalizing similar/repetitive
actions in action sequences.
"""
import json
from typing import Any, Dict, List

from openjudge.graders.agent.utils import (
    calculate_text_similarity,
    extract_action_observation_pairs,
)
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore


class ActionLoopDetectionGrader(BaseGrader):
    """
    Detect and penalize similar/repetitive actions in action sequences.
    This grader identifies when the agent performs similar actions by checking
    all pairs of actions for similarity and penalizing based on the proportion
    of similar action pairs found.
    Example:
        >>> import asyncio
        >>> grader = ActionLoopDetectionGrader(similarity_threshold=1.0)
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[...],
        ... ))
        >>> print(f"Loop detection score: {result.score}")
    """

    def __init__(
        self,
        similarity_threshold: float = 1.0,
    ):
        """
        Initialize ActionLoopDetectionGrader.

        Args:
            similarity_threshold (float): Threshold to consider actions as similar.
                                         Defaults to 1.0.
        """
        super().__init__(
            name="action_loop_detection",
            mode=GraderMode.POINTWISE,
            description="Detect and penalize repetitive actions in sequences",
        )
        self.similarity_threshold = similarity_threshold

    async def _aevaluate(
        self,
        messages: List[Dict[str, Any]],
    ) -> GraderScore:
        """
        Detect loops in action sequences by comparing all pairs of action signatures.
        Args:
            messages: List of message dicts containing agent interactions
        Returns:
            GraderScore: Loop detection score (1.0 = no loops, 0.0 = many loops)
        Example:
            >>> grader = ActionLoopDetectionGrader(similarity_threshold=1.0)
            >>> result = await grader.aevaluate(
            ...     messages=[...],
            ... )
            >>> print(f"Loop detection score: {result.score}")
        """
        messages = [msg.get("message", msg) for msg in messages]
        action_obs_pairs = extract_action_observation_pairs(messages)
        # Extract action signatures (function name + key arguments)
        action_signatures = []
        for action, _ in action_obs_pairs:
            function = action.get("function", {})
            signature = function.get("name", "")
            # Add key argument values to signature
            try:
                arguments = function.get("arguments", "")
                args = json.loads(arguments)
                # Sort keys for consistent comparison
                key_args = sorted(args.items())
                signature += ": " + ", ".join(f"{k}: {v}" for k, v in key_args)
            except (json.JSONDecodeError, AttributeError):
                # Keep format consistent with success case using colon separator
                raw_args = function.get("arguments", "")
                if raw_args:
                    signature += ": " + str(raw_args)
            action_signatures.append(signature)
        n = len(action_signatures)
        if n < 2:
            # No comparisons possible, return full score
            loop_score = 1.0
            similar_pair_count = 0
            total_pair_count = 0
            similar_pairs = []
        else:
            similar_pair_count = 0
            total_pair_count = 0
            similar_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    total_pair_count += 1
                    similarity = calculate_text_similarity(
                        action_signatures[i],
                        action_signatures[j],
                    )
                    if similarity >= self.similarity_threshold:
                        similar_pair_count += 1
                        similar_pairs.append(
                            (action_signatures[i], action_signatures[j], similarity),
                        )
            loop_score = 1.0 - (similar_pair_count / total_pair_count) if total_pair_count > 0 else 1.0
        return GraderScore(
            name=self.name,
            score=loop_score,
            reason=f"Loop detection: {similar_pair_count}/{total_pair_count} pairs are "
            f"similar (threshold={self.similarity_threshold})",
            metadata={
                "action_count": n,
                "similar_pair_count": similar_pair_count,
                "total_pair_count": total_pair_count,
                "similar_pairs": similar_pairs,
                "similarity_threshold": self.similarity_threshold,
                "signatures": action_signatures,
            },
        )
