# -*- coding: utf-8 -*-
"""Helper utilities for OpenJudge Studio."""

import asyncio
import base64
import json
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new loop if the current one is running
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)


def parse_json_safely(json_str: str, default: Any = None) -> Any:
    """Parse JSON string safely, returning default on error.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    if not json_str or not json_str.strip():
        return default
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default


def format_score_display(
    score: float,
    score_range: tuple[float, float] = (0, 1),
) -> tuple[str, str]:
    """Format score for display.

    Args:
        score: The score value
        score_range: (min, max) score range

    Returns:
        Tuple of (formatted_score, range_label)
    """
    min_score, max_score = score_range

    # Handle integer scores (1-5 range)
    if max_score == 5 and min_score == 1:
        return f"{int(score)}", "out of 5"

    # Handle 0-1 normalized scores
    if max_score == 1:
        return f"{score:.2f}", "out of 1.0"

    # Handle 0-10 scores
    if max_score == 10:
        return f"{score:.1f}", "out of 10"

    # Default formatting
    return f"{score:.2f}", f"out of {max_score}"


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_base64_to_bytes(base64_str: str) -> bytes:
    """Decode base64 string to bytes.

    Args:
        base64_str: Base64 encoded string

    Returns:
        Decoded bytes
    """
    return base64.b64decode(base64_str)


def get_image_format(filename: str) -> str:
    """Get image format from filename.

    Args:
        filename: Image filename

    Returns:
        Image format (jpeg, png, gif, webp)
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    format_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "webp": "webp",
    }
    return format_map.get(ext, "jpeg")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def validate_api_key(api_key: str) -> bool:
    """Basic validation for API key format.

    Args:
        api_key: API key string

    Returns:
        True if valid format
    """
    if not api_key:
        return False
    # Basic length check
    return len(api_key) >= 10


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time for display.

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
