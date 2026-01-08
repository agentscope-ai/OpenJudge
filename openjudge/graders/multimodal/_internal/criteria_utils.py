# -*- coding: utf-8 -*-
"""
Multimodal Custom Criteria Utility Functions

Helper functions for multimodal custom criteria evaluation.
"""

from typing import List, Optional, Tuple

from openjudge.graders.multimodal._internal.schema import Rubric

# Mapping for parameter display names (used in prompt generation)
PARAM_DISPLAY_NAMES = {
    "input": "Input",
    "actual_output": "Actual Output",
    "expected_output": "Expected Output",
    "context": "Context",
    "retrieval_context": "Retrieval Context",
    "tools": "Tools",
    "expected_tools": "Expected Tools",
}


def validate_criteria_and_evaluation_steps(
    criteria: Optional[str] = None,
    evaluation_steps: Optional[List[str]] = None,
) -> None:
    """
    Validate that either criteria or evaluation steps are provided

    Args:
        criteria: Evaluation criteria description
        evaluation_steps: List of evaluation steps

    Raises:
        ValueError: If neither or both are invalid

    Example:
        >>> validate_criteria_and_evaluation_steps(
        ...     criteria="Evaluate image quality"
        ... )  # OK
        >>> validate_criteria_and_evaluation_steps()  # Raises ValueError
    """
    # Check if both criteria and evaluation_steps are not None at the same time
    if criteria is None and evaluation_steps is None:
        raise ValueError("Either 'criteria' or 'evaluation_steps' must be provided.")

    # Check if criteria is provided, it cannot be an empty string
    if criteria is not None and not criteria.strip():
        raise ValueError("Criteria provided cannot be an empty string.")

    # Check if evaluation_steps is provided, it cannot be an empty list
    if evaluation_steps is not None and len(evaluation_steps) == 0:
        raise ValueError(
            "'evaluation_steps' must not be an empty list."
            "Either omit evaluation steps or include a non-empty list of steps.",
        )


# pylint: disable=unused-variable, consider-using-enumerate
def validate_and_sort_rubrics(
    rubrics: Optional[List[Rubric]] = None,
) -> Optional[List[Rubric]]:
    """
    Validate and sort rubrics by score range

    Args:
        rubrics: List of rubric definitions
        score_range: Valid score range (min, max)

    Returns:
        Sorted list of rubrics or None

    Raises:
        ValueError: If rubrics have overlapping ranges

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(7, 10), expected_outcome="Excellent"),
        ...     Rubric(score_range=(0, 6), expected_outcome="Poor")
        ... ]
        >>> sorted_rubrics = validate_and_sort_rubrics(rubrics)
    """
    if not rubrics:
        return None

    # Sort rubrics by start of range
    sorted_rubrics = sorted(rubrics, key=lambda r: r.score_range[0])

    # Full overlap check
    for i in range(len(sorted_rubrics)):
        a_start, a_end = sorted_rubrics[i].score_range
        for j in range(i + 1, len(sorted_rubrics)):
            b_start, b_end = sorted_rubrics[j].score_range
            # Check if ranges overlap
            if a_end >= b_start:
                raise ValueError(
                    f"Overlapping score ranges: {sorted_rubrics[i].score_range} and {sorted_rubrics[j].score_range}",
                )

    return sorted_rubrics


def format_rubrics(rubrics: Optional[List[Rubric]]) -> Optional[str]:
    """
    Format rubrics into a readable string

    Args:
        rubrics: List of rubric definitions

    Returns:
        Formatted rubric string or None

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(0, 3), expected_outcome="Poor quality"),
        ...     Rubric(score_range=(7, 10), expected_outcome="High quality")
        ... ]
        >>> print(format_rubrics(rubrics))
        0-3: Poor quality
        7-10: High quality
    """
    if not rubrics:
        return None

    return "\n".join(
        (f"{start}: {rubric.expected_outcome}" if start == end else f"{start}-{end}: {rubric.expected_outcome}")
        for rubric in rubrics
        for start, end in [rubric.score_range]
    )


def construct_params_string(
    evaluation_params: List[str],
) -> str:
    """
    Construct a readable string from evaluation parameters

    Args:
        evaluation_params: List of evaluation parameters

    Returns:
        Formatted parameter string

    Example:
        >>> params = ["input", "actual_output"]
        >>> construct_params_string(params)
        'Input and Actual Output'
    """
    params = [PARAM_DISPLAY_NAMES[param] for param in evaluation_params]

    if len(params) == 1:
        params_str = params[0]
    elif len(params) == 2:
        params_str = " and ".join(params)
    else:
        params_str = ", ".join(params[:-1]) + ", and " + params[-1]

    return params_str


def get_score_range(rubric: Optional[List[Rubric]]) -> Tuple[int, int]:
    """
    Get the overall score range from rubrics

    Args:
        rubric: List of rubric definitions

    Returns:
        Tuple of (min_score, max_score)

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(0, 3), expected_outcome="Poor"),
        ...     Rubric(score_range=(7, 10), expected_outcome="Excellent")
        ... ]
        >>> get_score_range(rubrics)
        (0, 10)
    """
    if not rubric:
        return (0, 10)

    return rubric[0].score_range[0], rubric[-1].score_range[1]


__all__ = [
    "validate_criteria_and_evaluation_steps",
    "validate_and_sort_rubrics",
    "format_rubrics",
    "construct_params_string",
    "get_score_range",
]
