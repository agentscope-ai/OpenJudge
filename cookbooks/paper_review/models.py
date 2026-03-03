# -*- coding: utf-8 -*-
"""LiteLLM-based model wrapper for PDF support."""

import base64
import os
from typing import Any, List, Optional

import litellm

litellm.drop_params = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


def _pdf_base64_to_text(data_url: str) -> str:
    """Extract text from a base64-encoded PDF data URL using PyMuPDF."""
    try:
        import pymupdf  # PyMuPDF

        if data_url.startswith("data:application/pdf;base64,"):
            b64 = data_url[len("data:application/pdf;base64,"):]
        else:
            b64 = data_url
        pdf_bytes = base64.b64decode(b64)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(pages_text)
    except Exception as e:
        return f"[PDF text extraction failed: {e}]"


def _transform_messages_for_text_api(messages: List[dict]) -> List[dict]:
    """Convert any 'file' content blocks (PDF) to plain text blocks."""
    transformed = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "file":
                    file_data = block.get("file", {}).get("file_data", "")
                    text = _pdf_base64_to_text(file_data)
                    new_content.append({"type": "text", "text": text})
                else:
                    new_content.append(block)
            transformed.append({**msg, "content": new_content})
        else:
            transformed.append(msg)
    return transformed


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

        try:
            response = litellm.completion(**completion_kwargs)
        except litellm.BadRequestError as e:
            if "file" in str(e).lower() or "invalid value" in str(e).lower():
                # API does not support 'file' type — convert PDF to text and retry
                completion_kwargs["messages"] = _transform_messages_for_text_api(messages)
                response = litellm.completion(**completion_kwargs)
            else:
                raise
        return _LiteLLMResponse(response.choices[0].message.content)

    def _get_model_name(self) -> str:
        """Get model name with provider prefix if needed."""
        model = self.model
        # Add provider prefix for litellm routing
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"
        elif "claude" in model.lower() and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        elif self.base_url and not any(model.startswith(p) for p in ("openai/", "anthropic/", "gemini/", "azure/", "bedrock/")):
            # When using a custom base_url with an OpenAI-compatible API (e.g. DashScope, vLLM),
            # LiteLLM requires the "openai/" prefix to route correctly.
            model = f"openai/{model}"
        return model


class _LiteLLMResponse:
    """Simple response wrapper."""

    def __init__(self, content: str):
        self.content = content
