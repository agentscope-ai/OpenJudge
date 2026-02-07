# -*- coding: utf-8 -*-
"""LiteLLM-based model wrapper for PDF support."""

import os
from typing import Any, List, Optional

import litellm

litellm.drop_params = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


class LiteLLMModel:
    """LiteLLM-based model with native PDF support."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 1500,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

    async def achat(self, messages: List[dict], **kwargs) -> Any:
        """Async chat completion with PDF support."""
        import asyncio

        return await asyncio.to_thread(self._chat_sync, messages, **kwargs)

    def _chat_sync(self, messages: List[dict], **kwargs) -> Any:
        """Sync chat completion."""
        completion_kwargs = {
            "model": self._get_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **kwargs,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.base_url:
            completion_kwargs["api_base"] = self.base_url

        response = litellm.completion(**completion_kwargs)
        return _LiteLLMResponse(response.choices[0].message.content)

    def _get_model_name(self) -> str:
        """Get model name with provider prefix if needed."""
        model = self.model
        # Add provider prefix for litellm routing
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"
        elif "claude" in model.lower() and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        return model


class _LiteLLMResponse:
    """Simple response wrapper."""

    def __init__(self, content: str):
        self.content = content
