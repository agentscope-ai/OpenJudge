# -*- coding: utf-8 -*-
"""MiniMax Chat Model."""

import os
import re
from typing import Any, AsyncGenerator, Callable, Dict, Literal, Type

from pydantic import BaseModel

from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse

# MiniMax-supported models with 204K context window
MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from MiniMax model output."""
    return _THINK_TAG_RE.sub("", text).strip()


class MiniMaxChatModel(OpenAIChatModel):
    """MiniMax chat model, using the OpenAI-compatible API at api.minimax.io.

    Supported models (all with 204K context window):
        - ``MiniMax-M2.7`` — latest generation, best quality
        - ``MiniMax-M2.7-highspeed`` — latest generation, faster inference
        - ``MiniMax-M2.5`` — previous generation
        - ``MiniMax-M2.5-highspeed`` — previous generation, faster inference

    .. note::
        MiniMax requires ``temperature`` to be in the range ``(0.0, 1.0]``.
        Values outside this range are automatically clamped.

    Example:
        >>> import asyncio, os
        >>> from openjudge.models import MiniMaxChatModel
        >>> from openjudge.graders.common.correctness import CorrectnessGrader
        >>>
        >>> model = MiniMaxChatModel(model="MiniMax-M2.7")
        >>> grader = CorrectnessGrader(model=model)
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     reference_response="The capital of France is Paris.",
        ... ))
        >>> print(result.score)
    """

    MINIMAX_BASE_URL = "https://api.minimax.io/v1"

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        client_args: Dict[str, Any] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MiniMax chat model.

        Args:
            model: MiniMax model name. Defaults to ``"MiniMax-M2.7"``.
                Available models: ``MiniMax-M2.7``, ``MiniMax-M2.7-highspeed``,
                ``MiniMax-M2.5``, ``MiniMax-M2.5-highspeed``.
            api_key: MiniMax API key. Falls back to the ``MINIMAX_API_KEY``
                environment variable.
            base_url: API base URL. Defaults to ``https://api.minimax.io/v1``.
            stream: Whether to use streaming output. Defaults to ``False``.
            client_args: Extra keyword arguments forwarded to :class:`openai.AsyncOpenAI`.
            max_retries: Number of retry attempts on transient errors.
            timeout: Request timeout in seconds.
            **kwargs: Extra keyword arguments forwarded to each API call
                (e.g. ``max_tokens``). ``temperature`` is clamped to ``(0, 1.0]``
                automatically.
        """
        resolved_api_key = api_key or os.getenv("MINIMAX_API_KEY")
        resolved_base_url = base_url or self.MINIMAX_BASE_URL

        # Clamp temperature to MiniMax's allowed range (0.0, 1.0]
        if "temperature" in kwargs:
            raw_temp = float(kwargs["temperature"])
            kwargs["temperature"] = max(1e-6, min(raw_temp, 1.0))

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

    async def achat(
        self,
        messages: list[dict | ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Chat with a MiniMax model.

        Wraps :meth:`OpenAIChatModel.achat` with MiniMax-specific adjustments:

        * ``temperature`` values outside ``(0.0, 1.0]`` are clamped.
        * ``<think>…</think>`` reasoning blocks produced by M2.5/M2.7 models
          are stripped from non-streaming responses so downstream graders always
          receive clean text.

        Args:
            messages: Conversation history as a list of message dicts or
                :class:`ChatMessage` objects.
            tools: Tool/function schemas available to the model.
            tool_choice: Tool selection strategy.
            structured_model: Pydantic model for structured output.
            callback: Optional callback invoked on the final response.
            **kwargs: Additional keyword arguments forwarded to the API.
                ``temperature`` is clamped to ``(0.0, 1.0]``.

        Returns:
            A :class:`ChatResponse` or an async generator thereof (streaming).
        """
        # Clamp temperature supplied at call-time as well
        if "temperature" in kwargs:
            raw_temp = float(kwargs["temperature"])
            kwargs["temperature"] = max(1e-6, min(raw_temp, 1.0))

        result = await super().achat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_model=structured_model,
            callback=callback,
            **kwargs,
        )

        # Strip <think>…</think> blocks from non-streaming responses
        if isinstance(result, ChatResponse):
            if result.content and isinstance(result.content, str):
                result.content = _strip_think_tags(result.content)

        return result


__all__ = ["MiniMaxChatModel", "MINIMAX_MODELS"]
