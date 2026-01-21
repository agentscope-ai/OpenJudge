# -*- coding: utf-8 -*-
"""
Tool Call Sequence Grader
This module provides graders for evaluating tool call sequences against
reference references, supporting both strict and loose matching modes.
"""
import json
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from openjudge.graders.base_grader import (
    BaseGrader,
    GraderError,
    GraderMode,
    GraderScore,
)

# pylint: disable=line-too-long


class ToolCallSequenceMatchGrader(BaseGrader):
    """
    Tool call sequence reference matching grader.
    This grader evaluates whether the model's tool call sequence matches the reference
    expected sequence by comparing predicted tool calls against reference tool calls.
    **Strict mode**: Matches both tool_call name and arguments, using F1 score calculation.
    **Loose mode**: Only matches tool_call name, checking whether model tool list is a subset of reference.
    Attributes:
        strict_mode: If True, matches both tool_call name and arguments; if False, only matches tool_call name
        use_jaccard_similarity: If True, use Jaccard similarity for loose mode (ignores step order)
    Example:
        >>> import asyncio
        >>> grader = ToolCallSequenceMatchGrader(strict_mode=True)
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[...],  # Model's messages with tool calls
        ...     reference_tool_calls=[...]  # Ground truth reference tool calls
        ... ))
        >>> print(f"Sequence match score: {result.score}")
    """

    def __init__(
        self,
        strict_mode: bool = True,
        use_jaccard_similarity: bool = True,
        metric_type: str = "recall",
        **kwargs,
    ):
        """
        Initialize the ToolCallSequenceMatchGrader.

        Args:
            strict_mode: If True, matches both tool_call name and arguments; if False, only matches tool_call name
            use_jaccard_similarity: If True, use Jaccard similarity for loose mode (ignores step order)
            metric_type: Metric type for step matching when use_jaccard_similarity=False and strict_mode=False.
                - "recall": matched_count / reference_count (default)
                - "precision": matched_count / predicted_count
            **kwargs: Additional arguments passed to BaseGrader
        """
        super().__init__(
            name="tool_call_sequence",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool call sequence matching against reference",
            **kwargs,
        )
        if metric_type not in ("recall", "precision"):
            raise ValueError(f"metric_type must be 'recall' or 'precision', got '{metric_type}'")
        self.strict_mode = strict_mode
        self.use_jaccard_similarity = use_jaccard_similarity
        self.metric_type = metric_type

    def extract_predicted_tool_sequence(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract the predicted tool call sequence from model messages, organized by steps.
        Args:
            messages: List of message dicts containing the model's messages, including tool calls
        Returns:
            A dictionary mapping step numbers to lists of tool call information within that step
        """
        step_tools = {}
        step_idx = 0
        for message in messages:
            role = message.get("role", "")
            tool_calls = message.get("tool_calls", [])
            # Process tool calls only from messages with role 'assistant'
            if role == "assistant" and tool_calls:
                step_tools[step_idx] = []
                for chat_tool in tool_calls:
                    # Parse the tool call arguments
                    chat_tool = chat_tool.get("tool_call", chat_tool)
                    function = chat_tool.get("function", chat_tool)
                    raw_args = function.get("arguments", "{}")

                    if isinstance(raw_args, dict):
                        params = raw_args
                    else:
                        try:
                            params = json.loads(raw_args)
                        except json.JSONDecodeError:
                            params = {}

                    # Prepare the tool information
                    tool_info = {
                        "name": function.get("name", ""),
                        "arguments": params,
                    }
                    step_tools[step_idx].append(tool_info)
                # Increment the step index for the next step
                step_idx += 1
        return step_tools

    def extract_reference_tool_sequence(
        self,
        reference_tool_calls: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract reference tool call sequence from reference tool calls, organized by steps.
        Args:
            reference_tool_calls: Ground truth tool call list in format:
        Returns:
            Dictionary mapping step numbers to lists of tool calls within that step
        """
        step_tools = {}
        step = 0
        for step_info in reference_tool_calls:
            tool_calls = step_info
            if step not in step_tools:
                step_tools[step] = []
            for tool_call in tool_calls:
                tool_call = tool_call.get("tool_call", tool_call)
                function = tool_call.get("function", tool_call)
                raw_args = function.get("arguments", "{}")

                if isinstance(raw_args, dict):
                    params = raw_args
                else:
                    try:
                        params = json.loads(raw_args)
                    except json.JSONDecodeError:
                        params = {}

                tool_info = {
                    "name": function.get("name", ""),
                    "arguments": params,
                }
                step_tools[step].append(tool_info)
            step += 1
        return step_tools

    def create_tool_elements(self, tool: Dict[str, Any]) -> Set[str]:
        """
        Create elements for tool matching based on matching mode.
        Args:
            tool: Tool information dictionary
        Returns:
            Set of elements representing the tool
        """
        elements = set()
        # Add flattened arguments
        params = tool.get("arguments", {})
        for key, value in params.items():
            elements.add(str(key))
            elements.add(str(value))
        return elements

    def calculate_param_similarity(
        self,
        ref_tool: Dict[str, Any],
        model_tool: Dict[str, Any],
    ) -> float:
        """
        Calculate F1 score for complete tool parameters matching.
        Args:
            ref_tool: Ground truth tool with name and parameters
            model_tool: Model tool with name and parameters
        Returns:
            Float similarity score between 0 and 1
        """
        ref_elements = self.create_tool_elements(ref_tool)
        model_elements = self.create_tool_elements(model_tool)
        # Handle edge cases
        if not ref_elements and not model_elements:
            return 1.0
        if not ref_elements or not model_elements:
            return 0.0
        # Calculate precision, recall, and F1
        true_positives = len(ref_elements & model_elements)
        precision = true_positives / len(model_elements) if model_elements else 0.0
        recall = true_positives / len(ref_elements) if ref_elements else 0.0
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def is_subsequence_unordered(
        self,
        list1: List[str],
        list2: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check if all elements of list1 are present in list2, regardless of order.
        Args:
            list1: Target list (reference)
            list2: Source list (model output)
        Returns:
            Tuple of (is_subsequence, missing_elements)
        """
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        missing_elements = []
        for elem, count in counter1.items():
            if counter2.get(elem, 0) < count:
                missing_elements.extend([elem] * (count - counter2.get(elem, 0)))
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements

    def calculate_step_matching_score(
        self,
        predicted_tool_steps: Dict[int, List[Dict[str, Any]]],
        reference_tool_steps: Dict[int, List[Dict[str, Any]]],
    ) -> float:
        """
        Calculate step matching score by comparing each step between predicted and reference tool calls.
        Uses F1 score for parameter matching in strict mode and improved scoring based on missing tools.
        Args:
            predicted_tool_steps: Model's predicted tool calls organized by steps
            reference_tool_steps: Ground truth reference tool calls organized by steps
        Returns:
            Step matching score (0.0 - 1.0) based on proportion of matched steps
        """
        if not reference_tool_steps:
            return 1.0 if not predicted_tool_steps else 0.0
        total_score = 0.0
        total_steps = len(reference_tool_steps)
        # Iterate through each step in reference_tool_steps
        for step_index, reference_tools in reference_tool_steps.items():
            step_score = 0.0
            # Check if model has the corresponding step
            if step_index in predicted_tool_steps:
                predicted_tools = predicted_tool_steps[step_index]
                # Extract tool names
                gt_tool_names = [tool.get("name", "") for tool in reference_tools]
                pred_tool_names = [tool.get("name", "") for tool in predicted_tools]
                # Use unordered matching to check tools in this step
                is_match, missing = self.is_subsequence_unordered(
                    gt_tool_names,
                    pred_tool_names,
                )
                if self.strict_mode:
                    if not is_match:
                        # Tool names don't match, score is 0
                        step_score = 0.0
                    else:
                        # Tool names match, calculate F1 score for parameters
                        # For each reference tool, find matching predicted tool by name and calculate F1
                        tool_scores = []
                        for gt_tool in reference_tools:
                            gt_name = gt_tool.get("name", "")
                            # Find matching predicted tool by name
                            matching_pred_tool = None
                            for pred_tool in predicted_tools:
                                if pred_tool.get("name", "") == gt_name:
                                    matching_pred_tool = pred_tool
                                    break

                            if matching_pred_tool is not None:
                                # Calculate F1 score for this tool pair
                                tool_f1 = self.calculate_param_similarity(
                                    gt_tool,
                                    matching_pred_tool,
                                )
                                tool_scores.append(tool_f1)
                            else:
                                # Ground truth tool not found in predictions, score is 0
                                tool_scores.append(0.0)

                        # Average F1 score across all reference tools in this step
                        step_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0
                else:
                    # In loose mode, calculate step score based on the ratio of matched tools
                    matched_count = len(gt_tool_names) - len(missing)
                    if self.metric_type == "recall":
                        # Recall: matched / reference
                        if len(gt_tool_names) > 0:
                            step_score = matched_count / len(gt_tool_names)
                        else:
                            step_score = 1.0
                    else:  # precision
                        # Precision: matched / predicted
                        if len(pred_tool_names) > 0:
                            step_score = matched_count / len(pred_tool_names)
                        else:
                            step_score = 0.0 if len(gt_tool_names) > 0 else 1.0
            else:
                step_score = 0.0  # No matching step in model
            total_score += step_score
        return total_score / total_steps if total_steps > 0 else 0.0

    def calculate_jaccard_similarity_score(
        self,
        predicted_tool_steps: Dict[int, List[Dict[str, Any]]],
        reference_tool_steps: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[float, Set[str], Set[str]]:
        """
        Calculate Jaccard similarity score for tool calls, ignoring step order.
        Treats all tool calls as a set and computes intersection over union.
        Args:
            predicted_tool_steps: Model's predicted tool calls organized by steps
            reference_tool_steps: Ground truth reference tool calls organized by steps
        Returns:
            Tuple of (jaccard_score, intersection_set, union_set)
        """
        if self.strict_mode:
            # In strict mode, we need to consider both tool names and parameters
            predicted_tool_names = set()
            for tools in predicted_tool_steps.values():
                for tool in tools:
                    predicted_tool_names.add(
                        f"{tool.get('name', '')}: {tool.get('arguments', {})}",
                    )
            reference_tool_names = set()
            for tools in reference_tool_steps.values():
                for tool in tools:
                    reference_tool_names.add(
                        f"{tool.get('name', '')}: {tool.get('arguments', {})}",
                    )
        else:
            # Extract all tool names from predicted steps
            predicted_tool_names = set()
            for tools in predicted_tool_steps.values():
                for tool in tools:
                    predicted_tool_names.add(tool.get("name", ""))
            # Extract all tool names from reference steps
            reference_tool_names = set()
            for tools in reference_tool_steps.values():
                for tool in tools:
                    reference_tool_names.add(tool.get("name", ""))
        # Handle edge cases
        if not reference_tool_names and not predicted_tool_names:
            return 1.0, set(), set()  # Both empty sets
        if not reference_tool_names or not predicted_tool_names:
            return 0.0, set(), set()  # One empty set
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection_set = predicted_tool_names & reference_tool_names
        union_set = predicted_tool_names | reference_tool_names
        score = len(intersection_set) / len(union_set) if union_set else 0.0
        return score, intersection_set, union_set

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
        reference_tool_calls: List[List[Dict[str, Any]]],
    ) -> GraderScore | GraderError:
        """
        Evaluate tool call sequence matching against reference.
        Args:
            messages: List of message dicts containing model's predicted tool calls
                "message" key for message, and "tool_call" key for tool call is optional.
                example without "message" and "tool_call"
                ```
                [
                  {"role": "system", "content": "..."},
                  {"role": "user", "content": "Plan travel from Shanghai to Hangzhou."},
                  {"role": "assistant", "tool_calls": [{"function": {"arguments": "{\"city\": \"Hangzhou\"}","name": "weather"}}]}
                ]
                ```
                or with "message" and "tool_call"
                ```
                [
                  {"message":{"role": "system", "content": "..."}},
                  {"message":{"role": "user", "content": "Plan travel from Shanghai to Hangzhou."}},
                  {"message":{"role": "assistant", "tool_calls": [{"tool_call":{"function": {"arguments": "{\"city\": \"Hangzhou\"}","name": "weather"}}}]}}
                ]
                ```
            reference_tool_calls: Ground truth reference tool call sequence by steps
                one step can contains a list of tool calls array.
                "tool_call" and "function" key for tool calls is optional.
                ```
                [
                    [
                        {
                            "tool_call": {"function": {"name": "search", "arguments": {...}}}
                        },
                        {
                            "tool_call": {"function": {"name": "search", "arguments": {...}}}
                        }
                    ]
                ]
                ```
                or, skip "tool_call" and "function"
                ```
                [
                    [
                        {"name": "search", "arguments": {...}},
                        {"name": "search", "arguments": {...}}
                    ]
                ]
                ```
        Returns:
            GraderScore: Tool call sequence matching score and details
        """
        messages = [msg.get("message", msg) for msg in messages]
        # Extract sequences
        try:
            predicted_tool_steps = self.extract_predicted_tool_sequence(messages)
            reference_tool_steps = self.extract_reference_tool_sequence(reference_tool_calls)
        except Exception as e:
            logger.error(f"Sequence extraction failed: {e}")
            return GraderError(
                name=self.name,
                error=f"Sequence extraction failed: {str(e)}",
                metadata={"error": str(e)},
            )
        if not predicted_tool_steps and not reference_tool_steps:
            return GraderScore(
                name=self.name,
                score=1.0,
                reason="Both predicted and reference have no tool calls",
                metadata={
                    "predicted_tool_count": 0,
                    "reference_tool_count": 0,
                },
            )
        # Calculate scores using the appropriate function
        if self.use_jaccard_similarity:
            (
                jaccard_score,
                intersection_set,
                union_set,
            ) = self.calculate_jaccard_similarity_score(predicted_tool_steps, reference_tool_steps)
            final_score = jaccard_score
            score_type = "jaccard_similarity"
        else:
            step_matching_score = self.calculate_step_matching_score(
                predicted_tool_steps,
                reference_tool_steps,
            )
            final_score = step_matching_score
            score_type = "step_matching"
        # Generate detailed reason
        mode_str = "strict" if self.strict_mode else "loose"
        method_str = "jaccard" if self.use_jaccard_similarity else f"step-by-step/{self.metric_type}"
        reason = f"Tool call sequence evaluation ({mode_str} mode, {method_str}): {score_type}={final_score:.3f}"
        # Count tools for metadata
        predicted_tool_count = sum(len(tools) for tools in predicted_tool_steps.values())
        reference_tool_count = sum(len(tools) for tools in reference_tool_steps.values())
        # Prepare metadata
        metadata = {
            "strict_mode": self.strict_mode,
            "use_jaccard_similarity": self.use_jaccard_similarity,
            "predicted_tool_count": predicted_tool_count,
            "reference_tool_count": reference_tool_count,
            "predicted_tool_steps": predicted_tool_steps,
            "reference_tool_steps": reference_tool_steps,
        }
        # Add the appropriate score to metadata
        if self.use_jaccard_similarity:
            metadata["jaccard_similarity_score"] = final_score
            metadata["intersection_set"] = list(intersection_set)
            metadata["union_set"] = list(union_set)
        else:
            metadata["step_matching_score"] = final_score
        return GraderScore(
            name=self.name,
            score=final_score,
            reason=reason,
            metadata=metadata,
        )
