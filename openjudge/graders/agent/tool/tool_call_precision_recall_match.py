# -*- coding: utf-8 -*-
"""
Simple Tool Call Sequence Grader
This module provides a simple grader for evaluating tool call sequences
by computing precision and recall metrics.
"""
import json
from enum import Enum
from typing import Any, Dict, List

from openjudge.graders.base_grader import (
    BaseGrader,
    GraderError,
    GraderMode,
    GraderScore,
)


class MetricType(str, Enum):
    """Enum for metric type selection."""

    PRECISION = "precision"
    RECALL = "recall"


class ToolCallPrecisionRecallMatchGrader(BaseGrader):
    """
    Simple tool call sequence matching grader.

    This grader evaluates tool calls by computing precision or recall metrics
    between predicted tool calls and reference tool calls.

    - **Precision**: The ratio of correctly predicted tool calls to total predicted tool calls.
    - **Recall**: The ratio of correctly predicted tool calls to total reference tool calls.

    Attributes:
        metric_type: Whether to return precision or recall score.
        match_arguments: If True, matches both tool name and arguments; if False, only matches tool name.

    Example:
        >>> import asyncio
        >>> grader = ToolCallPrecisionRecallMatchGrader(metric_type="precision", match_arguments=True)
        >>> result = asyncio.run(grader.aevaluate(
        ...     tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
        ...     reference_tool_calls=[{"name": "search", "arguments": {"query": "test"}}]
        ... ))
        >>> print(f"Precision score: {result.score}")
    """

    def __init__(
        self,
        metric_type: str = MetricType.RECALL,
        match_arguments: bool = False,
        **kwargs,
    ):
        """
        Initialize the simple tool call sequence match grader.

        Args:
            metric_type: The metric to compute. Either "precision" or "recall".
                - "precision": Correct predictions / Total predictions
                - "recall": Correct predictions / Total references
            match_arguments: If True, both tool name and arguments must match.
                If False, only tool name needs to match.
            **kwargs: Additional arguments passed to BaseGrader.
        """
        super().__init__(
            name="tool_call_sequence_simple",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool call precision/recall against reference",
            **kwargs,
        )
        # Convert string to enum if needed
        if isinstance(metric_type, str):
            self.metric_type = MetricType(metric_type)
        else:
            self.metric_type = metric_type
        self.match_arguments = match_arguments

    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a tool call to a standard format with 'name' and 'arguments'.

        Handles various input formats:
        - {"name": "...", "arguments": {...}}
        - {"function": {"name": "...", "arguments": {...}}}
        - {"tool_call": {"function": {"name": "...", "arguments": {...}}}}

        Args:
            tool_call: A tool call dict in any supported format.

        Returns:
            Normalized dict with 'name' and 'arguments' keys.
        """
        # Handle nested tool_call wrapper
        tool_call = tool_call.get("tool_call", tool_call)
        # Handle function wrapper
        function = tool_call.get("function", tool_call)

        name = function.get("name", "")
        raw_args = function.get("arguments", {})

        # Parse arguments if they are a JSON string
        if isinstance(raw_args, dict):
            arguments = raw_args
        else:
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}

        return {"name": name, "arguments": arguments}

    def _create_tool_key(self, tool: Dict[str, Any]) -> str:
        """
        Create a hashable key for a tool call based on matching mode.

        Args:
            tool: Normalized tool dict with 'name' and 'arguments'.

        Returns:
            A string key for the tool call.
        """
        if self.match_arguments:
            # Include arguments in the key for strict matching
            # Sort keys for consistent hashing
            args_str = json.dumps(tool.get("arguments", {}), sort_keys=True)
            return f"{tool.get('name', '')}:{args_str}"
        else:
            # Only use name for loose matching
            return tool.get("name", "")

    def _compute_metrics(
        self,
        tool_calls: List[Dict[str, Any]],
        reference_tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute precision and recall metrics.

        Args:
            tool_calls: List of predicted tool calls.
            reference_tool_calls: List of reference tool calls.

        Returns:
            Dict containing precision, recall, and match details.
        """
        # Normalize all tool calls
        predicted_tools = [self._normalize_tool_call(tc) for tc in tool_calls]
        reference_tools = [self._normalize_tool_call(tc) for tc in reference_tool_calls]

        # Create keys for comparison
        predicted_keys = [self._create_tool_key(t) for t in predicted_tools]
        reference_keys = [self._create_tool_key(t) for t in reference_tools]

        # Count matches (handle duplicates by counting occurrences)
        predicted_key_counts: Dict[str, int] = {}
        for key in predicted_keys:
            predicted_key_counts[key] = predicted_key_counts.get(key, 0) + 1

        reference_key_counts: Dict[str, int] = {}
        for key in reference_keys:
            reference_key_counts[key] = reference_key_counts.get(key, 0) + 1

        # Calculate true positives (intersection with counts)
        true_positives = 0
        matched_keys = []
        for key in set(predicted_keys) & set(reference_keys):
            match_count = min(predicted_key_counts[key], reference_key_counts[key])
            true_positives += match_count
            matched_keys.append(key)

        # Calculate precision and recall
        total_predicted = len(predicted_tools)
        total_reference = len(reference_tools)

        if total_predicted == 0:
            precision = 1.0 if total_reference == 0 else 0.0
        else:
            precision = true_positives / total_predicted

        if total_reference == 0:
            recall = 1.0 if total_predicted == 0 else 0.0
        else:
            recall = true_positives / total_reference

        return {
            "precision": precision,
            "recall": recall,
            "true_positives": true_positives,
            "total_predicted": total_predicted,
            "total_reference": total_reference,
            "matched_keys": matched_keys,
            "predicted_tools": predicted_tools,
            "reference_tools": reference_tools,
        }

    async def aevaluate(
        self,
        tool_calls: List[Dict[str, Any]],
        reference_tool_calls: List[Dict[str, Any]],
    ) -> GraderScore | GraderError:
        """
        Evaluate tool call precision/recall against reference.

        Args:
            tool_calls: List of predicted tool calls in format:
                ```
                [
                    {"name": "search", "arguments": {"query": "test"}},
                    {"name": "calculate", "arguments": {"expr": "1+1"}}
                ]
                ```
                or with nested format:
                ```
                [
                    {"function": {"name": "search", "arguments": {"query": "test"}}},
                    {"tool_call": {"function": {"name": "calculate", "arguments": {...}}}}
                ]
                ```

            reference_tool_calls: List of reference/ground truth tool calls in the same format.

        Returns:
            GraderScore: Contains the precision or recall score based on metric_type.
        """
        try:
            # Handle None inputs
            if tool_calls is None:
                tool_calls = []
            if reference_tool_calls is None:
                reference_tool_calls = []

            # Compute metrics
            metrics = self._compute_metrics(tool_calls, reference_tool_calls)

            # Select the appropriate score based on metric_type
            if self.metric_type == MetricType.PRECISION:
                score = metrics["precision"]
                score_name = "precision"
            else:
                score = metrics["recall"]
                score_name = "recall"

            # Generate reason
            match_mode = "strict (name + arguments)" if self.match_arguments else "loose (name only)"
            denominator = metrics["total_predicted"] if score_name == "precision" else metrics["total_reference"]
            reason = (
                f"Tool call {score_name}: {score:.3f} "
                f"({metrics['true_positives']}/{denominator}) "
                f"using {match_mode} matching"
            )

            return GraderScore(
                name=self.name,
                score=score,
                reason=reason,
                metadata={
                    "metric_type": self.metric_type.value,
                    "match_arguments": self.match_arguments,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "true_positives": metrics["true_positives"],
                    "total_predicted": metrics["total_predicted"],
                    "total_reference": metrics["total_reference"],
                    "matched_keys": metrics["matched_keys"],
                    "predicted_tools": metrics["predicted_tools"],
                    "reference_tools": metrics["reference_tools"],
                },
            )

        except Exception as e:
            return GraderError(
                name=self.name,
                error=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return information about the grader's evaluation process."""
        return {
            "description": "Simple tool call sequence matching grader using precision/recall metrics.",
            "parameters": {
                "metric_type": "Either 'precision' or 'recall'. "
                "Precision = TP / Total Predicted, Recall = TP / Total Reference.",
                "match_arguments": "If True, matches both tool name and arguments. "
                "If False, only matches tool name.",
            },
            "score_range": "0.0 to 1.0",
            "score_meaning": {
                "1.0": "Perfect match",
                "0.0": "No matches",
            },
        }
