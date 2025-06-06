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


class OpenaiLLM(BaseLLM):
    client: Any
    model: str = Field(default="gpt-4o")
    base_url: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    max_retries: int = Field(default=10)
    stream: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict):
        """Create an OpenAI client for Blt."""
        # Check for OPENAI_API_KEY
        openai_api_key = get_from_dict_or_env(
            data=data, key="OPENAI_API_KEY", default=None
        )
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it before using the client."
            )
        data["openai_api_key"] = openai_api_key
        data["base_url"] = get_from_dict_or_env(data, key="BASE_URL", default=None)

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
        call_params = {
            "model": self.model,
            # "top_p": self.top_p,
            "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
            "stream": self.stream,
        }

        # Remove None values
        call_params = {
            k: v
            for k, v in call_params.items()
            if v is not None and (isinstance(v, bool) or v != 0)
        }

        if self.tools:
            call_params.update({"tools": self.tools, "tool_choice": self.tool_choice})

        return call_params

    def chat(
        self, messages: List[ChatMessage] | str, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
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
        """Simple interface for chat with history support."""

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
        """Simple interface for chat with history support."""
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

            if len(ans) > 32768:
                return ans

        return ans
