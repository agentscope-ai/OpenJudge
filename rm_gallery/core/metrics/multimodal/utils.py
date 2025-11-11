"""
Utility functions for multimodal metrics
"""

import json
from typing import Any, Dict, List, Tuple, Union

from loguru import logger

from rm_gallery.core.metrics.multimodal.schema import MLLMImage


def separate_images_from_text(
    multimodal_list: List[Union[MLLMImage, str]]
) -> Tuple[List[str], List[MLLMImage]]:
    """
    Separate text and images from a multimodal list

    Args:
        multimodal_list: List containing both text and images

    Returns:
        Tuple of (text_list, image_list)

    Example:
        >>> texts, images = separate_images_from_text([
        ...     "Describe this:",
        ...     MLLMImage(url="..."),
        ...     "in detail"
        ... ])
        >>> len(texts)  # 2
        >>> len(images)  # 1
    """
    images: List[MLLMImage] = []
    texts: List[str] = []

    for item in multimodal_list:
        if isinstance(item, MLLMImage):
            images.append(item)
        elif isinstance(item, str):
            texts.append(item)

    return texts, images


def construct_verbose_logs(
    metric: Any,
    steps: List[str],
) -> str:
    """
    Construct verbose logs for a metric evaluation

    Args:
        metric: The metric instance
        steps: List of evaluation steps/information

    Returns:
        Formatted verbose log string

    Example:
        >>> logs = construct_verbose_logs(
        ...     metric,
        ...     steps=[
        ...         "Step 1: Extract features",
        ...         "Step 2: Compare similarity",
        ...         "Score: 0.85"
        ...     ]
        ... )
    """
    log_lines = [
        f"{'=' * 80}",
        f"Metric: {metric.name}",
        f"{'=' * 80}",
    ]

    for i, step in enumerate(steps, 1):
        log_lines.append(f"\nStep {i}:")
        log_lines.append(step)

    log_lines.append(f"\n{'=' * 80}")

    return "\n".join(log_lines)


def trim_and_load_json(response: str, metric: Any = None) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response

    Handles common cases where LLM wraps JSON in markdown code blocks or text.

    Args:
        response: LLM response string
        metric: Optional metric instance for error logging

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be parsed

    Example:
        >>> response = '''```json
        ... {"score": 8, "reasoning": "Good"}
        ... ```'''
        >>> data = trim_and_load_json(response)
        >>> data["score"]  # 8
    """
    # Remove markdown code blocks
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        error_msg = (
            f"Failed to parse JSON from response: {e}\nResponse: {response[:200]}"
        )
        if metric:
            logger.error(f"{metric.name}: {error_msg}")
        raise ValueError(error_msg)


def format_multimodal_content(
    content: List[Union[str, MLLMImage]], max_text_length: int = 500
) -> str:
    """
    Format multimodal content for display

    Args:
        content: List of text and images
        max_text_length: Maximum text length to display

    Returns:
        Formatted string representation

    Example:
        >>> formatted = format_multimodal_content([
        ...     "Long text...",
        ...     MLLMImage(url="https://example.com/img.jpg")
        ... ])
    """
    parts = []

    for item in content:
        if isinstance(item, str):
            if len(item) > max_text_length:
                parts.append(f"[Text: {item[:max_text_length]}...]")
            else:
                parts.append(f"[Text: {item}]")
        elif isinstance(item, MLLMImage):
            if item.url:
                parts.append(f"[Image: {item.url}]")
            else:
                parts.append("[Image: base64 data]")

    return " ".join(parts)


def calculate_score_from_reason_score(
    score: Union[int, float, List[int], List[float]], normalize: bool = True
) -> float:
    """
    Calculate normalized score from ReasonScore

    Args:
        score: Score value (single or list)
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Normalized score

    Example:
        >>> calculate_score_from_reason_score(8)  # 0.8
        >>> calculate_score_from_reason_score([8, 9])  # 0.85
    """
    if isinstance(score, list):
        avg_score = sum(score) / len(score) if score else 0.0
    else:
        avg_score = float(score)

    if normalize:
        # Assume scores are on 0-10 scale
        return avg_score / 10.0

    return avg_score


def validate_image_url(url: str) -> bool:
    """
    Validate if a string is a valid image URL

    Args:
        url: URL string to validate

    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False

    # Check if it starts with http/https
    if not (url.startswith("http://") or url.startswith("https://")):
        return False

    # Check for common image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    url_lower = url.lower()

    # Check if URL ends with image extension or has it before query params
    for ext in image_extensions:
        if ext in url_lower:
            return True

    # If no extension found, still return True (might be dynamic URL)
    return True


def merge_text_segments(
    multimodal_list: List[Union[str, MLLMImage]], separator: str = " "
) -> List[Union[str, MLLMImage]]:
    """
    Merge consecutive text segments in a multimodal list

    Args:
        multimodal_list: List containing text and images
        separator: Separator to use when merging text

    Returns:
        List with merged text segments

    Example:
        >>> merged = merge_text_segments([
        ...     "Hello",
        ...     "world",
        ...     MLLMImage(url="..."),
        ...     "foo",
        ...     "bar"
        ... ])
        >>> len(merged)  # 3: "Hello world", image, "foo bar"
    """
    if not multimodal_list:
        return []

    result = []
    current_text_segments = []

    for item in multimodal_list:
        if isinstance(item, str):
            current_text_segments.append(item)
        elif isinstance(item, MLLMImage):
            # Merge accumulated text segments
            if current_text_segments:
                merged_text = separator.join(current_text_segments)
                result.append(merged_text)
                current_text_segments = []
            # Add the image
            result.append(item)

    # Merge any remaining text segments
    if current_text_segments:
        merged_text = separator.join(current_text_segments)
        result.append(merged_text)

    return result
