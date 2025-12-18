# -*- coding: utf-8 -*-
"""
Utility Functions for Multimodal Graders

This module contains core data structures for multimodal graders.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MLLMImage(BaseModel):
    """
    Multimodal LLM Image representation

    Supports both URL-based and base64-encoded images.

    Attributes:
        url: Image URL (HTTP/HTTPS)
        base64: Base64-encoded image data
        format: Image format (png, jpg, jpeg, etc.)

    Example:
        >>> # URL-based image
        >>> img1 = MLLMImage(url="https://example.com/image.jpg")
        >>>
        >>> # Base64-encoded image
        >>> img2 = MLLMImage(
        ...     base64="iVBORw0KGgoAAAANS...",
        ...     format="png"
        ... )
    """

    url: Optional[str] = Field(None, description="Image URL")
    base64: Optional[str] = Field(None, description="Base64-encoded image data")
    format: Optional[str] = Field(None, description="Image format (png, jpg, etc.)")

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one of url or base64 is provided"""
        if not self.url and not self.base64:
            raise ValueError("Either 'url' or 'base64' must be provided")


def format_image_content(
    text: str,
    images: List[MLLMImage],
) -> List[Dict[str, Any]]:
    """
    Format text and images into OpenAI content format

    Args:
        text: Text content
        images: List of MLLMImage objects

    Returns:
        List of content dictionaries in OpenAI format

    Example:
        >>> content = format_image_content(
        ...     "Describe this image:",
        ...     [MLLMImage(url="https://example.com/image.jpg")]
        ... )
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

    for img in images:
        if img.url:
            content.append({"type": "image_url", "image_url": {"url": img.url}})
        elif img.base64:
            img_format = img.format or "jpeg"
            data_uri = f"data:image/{img_format};base64,{img.base64}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

    return content
