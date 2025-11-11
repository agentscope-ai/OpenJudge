"""
Utility functions for multimodal reward models
"""

from typing import List, Optional, Union

from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCase,
    MLLMTestCaseParams,
)
from rm_gallery.gallery.rm.multimodal.schema import Rubric


def construct_g_eval_params_string(evaluation_params: List[MLLMTestCaseParams]) -> str:
    """
    Construct a human-readable string from test case parameters

    Args:
        evaluation_params: List of test case parameters

    Returns:
        str: Human-readable parameter string

    Example:
        >>> params = [MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT]
        >>> construct_g_eval_params_string(params)
        'input and actual output'
    """
    if not evaluation_params:
        return "the test case"

    # Convert enum values to readable strings
    param_names = []
    for param in evaluation_params:
        name = param.value.replace("_", " ")
        param_names.append(name)

    # Join with proper grammar
    if len(param_names) == 1:
        return param_names[0]
    elif len(param_names) == 2:
        return f"{param_names[0]} and {param_names[1]}"
    else:
        return ", ".join(param_names[:-1]) + f", and {param_names[-1]}"


def construct_test_case_list(
    test_case: MLLMTestCase, evaluation_params: List[MLLMTestCaseParams]
) -> List[Union[str, MLLMImage]]:
    """
    Construct a list of test case components for evaluation

    Extracts the specified parameters from test case and formats them
    for inclusion in the evaluation prompt.

    Args:
        test_case: Test case to extract from
        evaluation_params: Which parameters to include

    Returns:
        List: Mixed list of text and images

    Example:
        >>> test_case = MLLMTestCase(
        ...     input=["Describe:", MLLMImage(url="...")],
        ...     actual_output=["A cat"]
        ... )
        >>> params = [MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT]
        >>> result = construct_test_case_list(test_case, params)
    """
    result = []

    for param in evaluation_params:
        param_value = getattr(test_case, param.value, None)

        if param_value is None:
            continue

        # Add section header
        param_name = param.value.replace("_", " ").title()
        result.append(f"\n{param_name}:")

        # Add content
        if isinstance(param_value, list):
            # Mixed list of text and images
            for item in param_value:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, MLLMImage):
                    result.append(item)
        elif isinstance(param_value, str):
            result.append(param_value)
        else:
            result.append(str(param_value))

    return result


def format_rubrics(rubrics: Optional[List[Rubric]]) -> Optional[str]:
    """
    Format rubrics into a readable string

    Args:
        rubrics: List of rubric entries

    Returns:
        str: Formatted rubric string, or None if no rubrics

    Example:
        >>> rubrics = [
        ...     Rubric(score=10, description="Perfect"),
        ...     Rubric(score=5, description="Average"),
        ...     Rubric(score=0, description="Poor")
        ... ]
        >>> format_rubrics(rubrics)
        'Score 0: Poor\\nScore 5: Average\\nScore 10: Perfect'
    """
    if not rubrics:
        return None

    lines = []
    for rubric in rubrics:
        lines.append(f"Score {rubric.score}: {rubric.description}")

    return "\n".join(lines)


def validate_and_sort_rubrics(
    rubrics: Optional[List[Rubric]],
) -> Optional[List[Rubric]]:
    """
    Validate and sort rubrics by score

    Args:
        rubrics: List of rubric entries

    Returns:
        List[Rubric]: Sorted rubrics, or None if input is None

    Raises:
        ValueError: If rubrics are invalid

    Example:
        >>> rubrics = [
        ...     Rubric(score=10, description="Perfect"),
        ...     Rubric(score=0, description="Poor"),
        ...     Rubric(score=5, description="Average")
        ... ]
        >>> sorted_rubrics = validate_and_sort_rubrics(rubrics)
        >>> [r.score for r in sorted_rubrics]
        [0, 5, 10]
    """
    if rubrics is None:
        return None

    if not rubrics:
        raise ValueError("Rubric list cannot be empty")

    # Check for duplicate scores
    scores = [r.score for r in rubrics]
    if len(scores) != len(set(scores)):
        raise ValueError("Rubric scores must be unique")

    # Sort by score
    return sorted(rubrics, key=lambda r: r.score)


def validate_criteria_and_evaluation_steps(
    criteria: Optional[str], evaluation_steps: Optional[List[str]]
) -> None:
    """
    Validate that either criteria or evaluation steps are provided

    Args:
        criteria: Evaluation criteria
        evaluation_steps: Evaluation steps

    Raises:
        ValueError: If neither criteria nor evaluation_steps are provided

    Example:
        >>> validate_criteria_and_evaluation_steps(
        ...     criteria="Evaluate quality",
        ...     evaluation_steps=None
        ... )  # OK
        >>> validate_criteria_and_evaluation_steps(
        ...     criteria=None,
        ...     evaluation_steps=["Step 1", "Step 2"]
        ... )  # OK
        >>> validate_criteria_and_evaluation_steps(
        ...     criteria=None,
        ...     evaluation_steps=None
        ... )  # Raises ValueError
    """
    if not criteria and not evaluation_steps:
        raise ValueError(
            "Either 'criteria' or 'evaluation_steps' must be provided. "
            "Criteria is used to auto-generate evaluation steps if not explicitly provided."
        )

    if evaluation_steps is not None and len(evaluation_steps) == 0:
        raise ValueError("evaluation_steps cannot be an empty list")
