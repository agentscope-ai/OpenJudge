# -*- coding: utf-8 -*-
"""Qiniu Chat Model."""

import os
from typing import Any, Dict

from openjudge.models.openai_chat_model import OpenAIChatModel

# Qiniu-supported default model list
QINIU_MODELS = [
    "deepseek-v3",
]


class QiniuChatModel(OpenAIChatModel):
    """Qiniu chat model using the OpenAI-compatible API gateway."""

    QINIU_BASE_URL = "https://api.qnaigc.com/v1"

    def __init__(
        self,
        model: str = "deepseek-v3",
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        client_args: Dict[str, Any] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the qiniu chat model."""
        resolved_api_key = api_key or os.getenv("QINIU_API_KEY")
        resolved_base_url = base_url or self.QINIU_BASE_URL

        super().__init__(
            model=model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            stream=stream,
            client_args=client_args,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )


__all__ = ["QiniuChatModel", "QINIU_MODELS"]
