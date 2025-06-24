from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from pydantic import Field, model_validator

from rm_gallery.core.model.base import (
    BaseLLM,
    _convert_chat_message_to_openai_message,
    _convert_openai_response_to_response,
    _convert_stream_chunk_to_response,
    get_from_dict_or_env,
)
from rm_gallery.core.model.message import (
    ChatMessage,
    ChatResponse,
    GeneratorChatResponse,
)
from rm_gallery.core.utils.text import detect_consecutive_repetition


class OpenaiLLM(BaseLLM):
    """
    OpenAI Language Model interface for chat completions with streaming and history support.

    Attributes:
        client (Any): OpenAI API client instance
        model (str): Model name/version to use
        base_url (str | None): Custom API endpoint URL
        openai_api_key (str | None): Authentication token
        max_retries (int): Maximum retry attempts for failed requests
        stream (bool): Whether to stream responses incrementally
        max_tokens (int): Maximum output tokens per response
    """

    client: Any
    model: str = Field(default="gpt-4o")
    base_url: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    max_retries: int = Field(default=10)
    stream: bool = Field(default=False)
    max_tokens: int = Field(default=8192)
    thinking_budget: int = Field(default=8192)
    stop_if_detect_repetition: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict):
        """
        Initialize and validate OpenAI client configuration.

        Ensures API key is available, then creates a configured OpenAI client instance.
        Handles environment variable fallback for configuration parameters.

        Args:
            data (Dict): Configuration dictionary containing potential client parameters

        Returns:
            Dict: Updated configuration with initialized client and validated parameters

        Raises:
            ValueError: If API key is missing or client initialization fails
        """
        # Check for OPENAI_API_KEY
        openai_api_key = get_from_dict_or_env(
            data=data, key="openai_api_key", default=None
        )
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it before using the client."
            )
        data["openai_api_key"] = openai_api_key
        data["base_url"] = get_from_dict_or_env(data, key="base_url", default=None)

        try:
            data["client"] = OpenAI(
                api_key=data["openai_api_key"],
                base_url=data["base_url"],
                max_retries=data.get("max_retries", 10),
                timeout=60.0,
            )
            return data
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        """
        Generate filtered keyword arguments for chat completion API calls.

        Includes model parameters with special handling for tool calls.
        Filters out None values and zero/false values except for boolean flags.

        Returns:
            Dict[str, Any]: Cleaned dictionary of chat completion parameters
        """
        call_params = {
            "model": self.model,
            # "top_p": self.top_p,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }

        # Remove None values
        call_params = {
            k: v
            for k, v in call_params.items()
            if v is not None and (isinstance(v, bool) or v != 0)
        }

        if "qwen3" in self.model:
            call_params["extra_body"] = {
                "enable_thinking": self.enable_thinking,
                "thinking_budget": self.thinking_budget,
            }

        if self.tools:
            call_params.update({"tools": self.tools, "tool_choice": self.tool_choice})

        return call_params

    def chat(
        self, messages: List[ChatMessage] | str, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
        """
        Process chat messages and generate responses from OpenAI API.

        Handles both single message and streaming response modes based on configuration.
        Converts messages to OpenAI format before making API call.

        Args:
            messages (List[ChatMessage] | str): Input messages in application format
            **kwargs: Additional parameters to override default chat settings

        Returns:
            ChatResponse | GeneratorChatResponse: Either complete response or streaming generator

        Raises:
            Exception: Wraps and re-raises API call failures
        """
        messages = self._convert_messages(messages)

        call_params = self.chat_kwargs.copy()
        call_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                messages=_convert_chat_message_to_openai_message(messages),
                **call_params,
            )

            if self.stream:
                return self._handle_stream_response(response)
            return _convert_openai_response_to_response(response)

        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def _handle_stream_response(self, response: Any) -> GeneratorChatResponse:
        """
        Process streaming response chunks into application format.

        Combines incremental response chunks while maintaining complete message context.
        Yields updated responses as new content becomes available.

        Args:
            response (Any): Raw streaming response from OpenAI API

        Yields:
            GeneratorChatResponse: Incremental updates to the complete response
        """
        _response = None
        for chunk in response:
            chunk_response = _convert_stream_chunk_to_response(chunk)
            if chunk_response is None:
                continue

            if _response is None:
                _response = chunk_response
            else:
                _response.message = _response.message + chunk_response.message
                _response.delta = chunk_response.message

            yield _response

    def simple_chat(
        self,
        query: str,
        history: Optional[List[str]] = None,
        sys_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> Any:
        """
        Simplified chat interface with built-in history management.

        Handles conversation history formatting and system prompt integration.
        Switches between reasoning and standard modes based on configuration.

        Args:
            query (str): Current user input text
            history (Optional[List[str]]): Previous conversation history
            sys_prompt (str): System-level instructions for the model
            **kwargs: Additional parameters for chat configuration

        Returns:
            Any: Model response content, typically a string
        """
        if self.enable_thinking:
            return self.simple_chat_reasoning(
                query=query, history=history, sys_prompt=sys_prompt, **kwargs
            )

        messages = [{"role": "system", "content": sys_prompt}]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = "user" if i % 2 == 0 else "assistant"
            messages += [{"role": role, "content": h}]

        call_params = self.chat_kwargs.copy()
        call_params.update(kwargs)
        response = self.client.chat.completions.create(messages=messages, **call_params)
        return _convert_openai_response_to_response(response).message.content

    def simple_chat_reasoning(
        self,
        query: str,
        history: Optional[List[str]] = None,
        sys_prompt: str = "",
        **kwargs,
    ) -> Any:
        """
        Enhanced chat interface with reasoning content processing.

        Handles special reasoning content markers and separates thinking from output.
        Implements token limit safety with early return for long responses.

        Args:
            query (str): User input text
            history (Optional[List[str]]): Conversation history
            sys_prompt (str): System instructions
            **kwargs: Chat configuration parameters

        Returns:
            Any: Combined response content with reasoning markers
        """
        messages = [{"role": "system", "content": sys_prompt}]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = "user" if i % 2 == 0 else "assistant"
            messages += [{"role": role, "content": h}]

        call_params = self.chat_kwargs.copy()
        call_params["stream"] = True
        call_params.update(kwargs)

        try:
            completion = self.client.chat.completions.create(
                messages=messages, **call_params
            )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            completion = self.client.chat.completions.create(
                messages=messages, **call_params
            )

        ans = ""
        enter_think = False
        leave_think = False
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if (
                    hasattr(delta, "reasoning_content")
                    and delta.reasoning_content is not None
                ):
                    if not enter_think:
                        enter_think = True
                        ans += "<think>"
                    ans += delta.reasoning_content
                elif delta.content:
                    if enter_think and not leave_think:
                        leave_think = True
                        ans += "</think>"
                    ans += delta.content
            if self.stop_if_detect_repetition:
                repetition_text = detect_consecutive_repetition(ans)
                if repetition_text:
                    logger.info(f"repetition_text={repetition_text},stop")
            if len(ans) > 32768:
                return ans

        return ans
