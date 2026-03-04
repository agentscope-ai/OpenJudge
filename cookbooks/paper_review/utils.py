# -*- coding: utf-8 -*-
"""Utility functions for paper review."""

import base64
from pathlib import Path
from typing import Any, Union


async def extract_response_content(response: Any) -> str:
    """Extract content from a model response, handling streaming responses.

    This helper handles both regular and streaming (async iterator) responses
    from chat models, providing a unified interface for response content extraction.

    Args:
        response: The response from model.achat(), which may be a regular
                  response object or an async iterator for streaming.

    Returns:
        The extracted content string from the response.
    """
    # Handle streaming responses
    if hasattr(response, "__aiter__"):
        async for chunk in response:
            response = chunk

    # Extract content from response
    return response.content if hasattr(response, "content") else str(response)


def load_pdf_bytes(pdf_path: Union[str, Path]) -> bytes:
    """Load PDF file as bytes."""
    with open(pdf_path, "rb") as f:
        return f.read()


def encode_pdf_base64(pdf_bytes: bytes) -> str:
    """Encode PDF bytes to base64 data URL for multimodal models."""
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def encode_image_base64(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Encode image bytes to base64 data URL."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
