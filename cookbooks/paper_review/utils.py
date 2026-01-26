# -*- coding: utf-8 -*-
"""Utility functions for paper review."""

import base64
from pathlib import Path
from typing import Union


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
