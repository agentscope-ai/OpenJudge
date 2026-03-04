# -*- coding: utf-8 -*-
"""Model factory for creating OpenJudge model instances."""

from typing import Any, Optional

from openjudge.models.openai_chat_model import OpenAIChatModel


def create_model(
    api_key: str,
    base_url: Optional[str] = None,
    model_name: str = "qwen3-32b",
    **extra_params: Any,
) -> OpenAIChatModel:
    """Create an OpenAIChatModel instance.

    Args:
        api_key: API key for the LLM service
        base_url: Optional custom API endpoint URL
        model_name: Name of the model to use
        **extra_params: Additional parameters to pass to the model

    Returns:
        Configured OpenAIChatModel instance
    """
    return OpenAIChatModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url if base_url else None,
        **extra_params,
    )
