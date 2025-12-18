# -*- coding: utf-8 -*-
"""
Context Extraction Utilities

Functions for extracting and processing context from multimodal content.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from rm_gallery.core.models.schema.qwen.mllmImage import MLLMImage


def get_image_indices(content_list: List[Union[str, "MLLMImage"]]) -> List[int]:
    """
    Find indices of all images in a content list

    Args:
        content_list: List containing text and images

    Returns:
        List of indices where images appear

    Example:
        >>> indices = get_image_indices([
        ...     "Text",
        ...     MLLMImage(url="..."),
        ...     "More text",
        ...     MLLMImage(url="...")
        ... ])
        >>> indices  # [1, 3]
    """
    from rm_gallery.core.models.schema.qwen.mllmImage import MLLMImage

    return [index for index, element in enumerate(content_list) if isinstance(element, MLLMImage)]


def get_image_context(
    image_index: int,
    content_list: List[Union[str, "MLLMImage"]],
    max_context_size: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text context surrounding an image

    Args:
        image_index: Index of the image in content_list
        content_list: List containing text and images
        max_context_size: Maximum characters to extract from context

    Returns:
        Tuple of (context_above, context_below)

    Example:
        >>> context_above, context_below = get_image_context(
        ...     1,
        ...     ["Sales data for Q3:", MLLMImage(url="..."), "Shows 20% growth"],
        ...     max_context_size=500
        ... )
    """
    context_above = None
    context_below = None

    # Find context above (last text before image)
    for i in range(image_index - 1, -1, -1):
        if isinstance(content_list[i], str):
            context_above = content_list[i]
            if max_context_size and len(context_above) > max_context_size:
                context_above = context_above[-max_context_size:]
            break

    # Find context below (first text after image)
    for i in range(image_index + 1, len(content_list)):
        if isinstance(content_list[i], str):
            context_below = content_list[i]
            if max_context_size and len(context_below) > max_context_size:
                context_below = context_below[:max_context_size]
            break

    return context_above, context_below
