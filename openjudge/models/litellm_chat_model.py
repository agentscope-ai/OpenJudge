# -*- coding: utf-8 -*-
"""LiteLLM chat model.

LiteLLM exposes 100+ LLM providers behind a single OpenAI-compatible
interface, routed by the model-name prefix (e.g. ``anthropic/claude-sonnet-4-6``,
``gemini/gemini-2.5-flash``, ``bedrock/...``). Because LiteLLM returns
OpenAI-shaped responses, this model reuses :class:`OpenAIChatModel`'s response
parsing (non-streaming and streaming) and only swaps the transport from the
OpenAI SDK to ``litellm.acompletion``.

Calling the SDK directly (rather than pointing ``OpenAIChatModel`` at a base
URL) lets LiteLLM use each provider's native authentication (Bedrock SigV4,
Vertex ADC, Azure AD) instead of only an OpenAI-style bearer token.
"""

from typing import Any, AsyncGenerator, Callable, Dict, Literal, Type

from loguru import logger
from pydantic import BaseModel

from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse


class LiteLLMChatModel(OpenAIChatModel):
    """Chat model backed by the LiteLLM SDK.

    Reuses :class:`OpenAIChatModel`'s OpenAI-shaped response handling and routes
    generation through ``litellm.acompletion``, so a single class reaches any
    LiteLLM-supported provider selected by the ``model`` prefix.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        drop_params: bool = True,
        max_retries: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LiteLLM chat model.

        Args:
            model: The LiteLLM model id, including its provider prefix where
                required (e.g. ``gpt-4o``, ``anthropic/claude-sonnet-4-6``,
                ``gemini/gemini-2.5-flash``).
            api_key: Optional API key. When omitted, LiteLLM falls back to each
                target provider's own environment variable (``OPENAI_API_KEY``,
                ``ANTHROPIC_API_KEY``, ...). Set it (with ``base_url``) to target
                a LiteLLM proxy.
            base_url: Optional API base, forwarded to LiteLLM as ``api_base``
                (e.g. a LiteLLM proxy URL).
            stream: Whether to stream the model output.
            reasoning_effort: Reasoning effort for models that support it.
            drop_params: Drop per-provider-unsupported params instead of raising,
                so one call shape works across providers. Defaults to ``True``.
            max_retries: Optional retry count, forwarded to LiteLLM as
                ``num_retries``.
            timeout: Optional request timeout in seconds.
            kwargs: Extra generation kwargs forwarded to ``litellm.acompletion``
                (e.g. ``temperature``, ``top_p``).
        """
        # Set the base attributes directly rather than calling
        # OpenAIChatModel.__init__, which builds a persistent AsyncOpenAI
        # client: LiteLLM has no persistent client and routes per call.
        self.model = model
        self.stream = stream
        self.reasoning_effort = reasoning_effort
        self.kwargs = kwargs or {}
        self.drop_params = drop_params
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    async def achat(
        self,
        messages: list[dict | ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Get a response from LiteLLM for the given arguments.

        The parameters mirror :meth:`OpenAIChatModel.achat`; see that method for
        details on ``tools``, ``tool_choice``, ``structured_model`` and
        ``callback``.

        Returns:
            Either a single :class:`ChatResponse` or, when ``stream`` is set, an
            async generator of :class:`ChatResponse` chunks.
        """
        import litellm

        messages = self._normalize_and_validate_messages(messages, "LiteLLM")

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            # Drop params a given provider does not support instead of erroring,
            # so the same request shape works across every backend.
            "drop_params": self.drop_params,
            **self.kwargs,
            **kwargs,
        }
        if self.reasoning_effort and "reasoning_effort" not in call_kwargs:
            call_kwargs["reasoning_effort"] = self.reasoning_effort

        # Forward credentials only when set, so LiteLLM otherwise falls back to
        # each provider's own env var; setting them targets a LiteLLM proxy.
        if self.api_key:
            call_kwargs.setdefault("api_key", self.api_key)
        if self.base_url:
            call_kwargs.setdefault("api_base", self.base_url)
        if self.max_retries is not None:
            call_kwargs.setdefault("num_retries", self.max_retries)
        if self.timeout is not None:
            call_kwargs.setdefault("timeout", self.timeout)

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and 'tool_choice' parameters will be "
                    "overridden and ignored. The model will only perform structured output generation.",
                )
            call_kwargs.pop("tools", None)
            call_kwargs.pop("tool_choice", None)

            # Some providers reject a full JSON schema; use a simple json_object.
            # deepseek included: it returns 400 "This response_format type is unavailable now".
            if any(name in self.model.lower() for name in ("qwen", "gemini", "pai-judge", "deepseek")):
                logger.info(
                    f"Model '{self.model}' detected: switching to 'json_object' response_format for compatibility"
                )
                call_kwargs["response_format"] = {"type": "json_object"}
            else:
                call_kwargs["response_format"] = structured_model
        else:
            if tools:
                call_kwargs["tools"] = tools
            if tool_choice:
                self._validate_tool_choice(tool_choice, tools)
                call_kwargs["tool_choice"] = tool_choice

        response = await litellm.acompletion(**call_kwargs)

        if call_kwargs["stream"]:
            return self._handle_streaming_response(response, structured_model, callback)
        return self._handle_non_streaming_response(response, structured_model, callback)
