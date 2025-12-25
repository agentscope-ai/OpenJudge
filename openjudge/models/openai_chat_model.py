# -*- coding: utf-8 -*-
"""OpenAI Client."""
import os
from typing import Any, AsyncGenerator, Callable, Dict, Literal, Type

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse
from openjudge.utils.utils import repair_and_load_json


def _format_audio_data_for_qwen_omni(messages: list[dict | ChatMessage]) -> None:
    """Qwen-omni uses OpenAI-compatible API but requires different audio
    data format than OpenAI with "data:;base64," prefix.
    Refer to `Qwen-omni documentation
    <https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2867839>`_
    for more details.

    Args:
        messages (`list[dict]`):
            The list of message dictionaries from OpenAI formatter.
    """
    for msg in messages:
        msg_dict = msg.to_dict() if isinstance(msg, ChatMessage) else msg
        if isinstance(msg_dict.get("content"), list):
            for block in msg_dict["content"]:
                if (
                    isinstance(block, dict)
                    and "input_audio" in block
                    and isinstance(block["input_audio"].get("data"), str)
                ):
                    if not block["input_audio"]["data"].startswith("http"):
                        block["input_audio"]["data"] = "data:;base64," + block["input_audio"]["data"]


class OpenAIChatModel(BaseChatModel):
    """The OpenAI chat model class, following AgentScope implementation."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        organization: str | None = None,
        client_args: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the openai client.

        Args:
            model: The name of the model to use in OpenAI API.
            api_key: The API key for OpenAI API. If not specified, it will
                be read from the environment variable `OPENAI_API_KEY`.
            base_url: The base URL for OpenAI API. If not specified, it will
                be read from the environment variable `OPENAI_BASE_URL`.
            stream: Whether to use streaming output or not.
            reasoning_effort: Reasoning effort, supported for o3, o4, etc.
            organization: The organization ID for OpenAI API. If not specified, it will
                be read from the environment variable `OPENAI_ORGANIZATION`.
            client_args: The extra keyword arguments to initialize the OpenAI client.
            kwargs: The extra keyword arguments used in OpenAI API generation,
                e.g. `temperature`, `seed`.
        """
        self.model = model
        self.stream = stream
        self.reasoning_effort = reasoning_effort
        self.kwargs = kwargs or {}

        # Initialize client
        client_args = client_args or {}
        if api_key:
            client_args["api_key"] = api_key
        else:
            client_args["api_key"] = os.getenv("OPENAI_API_KEY")

        if base_url:
            client_args["base_url"] = base_url
        else:
            client_args["base_url"] = os.getenv("OPENAI_BASE_URL", None)

        if organization:
            client_args["organization"] = organization

        self.client = AsyncOpenAI(**client_args)

    async def achat(
        self,
        messages: list[dict | ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Get the response from OpenAI chat completions API by the given
        arguments.

        Args:
            messages (`list[dict]`):
                A list of dictionaries, where `role` and `content` fields are
                required, and `name` field is optional.
            tools (`list[dict]`, default `None`):
                The tools JSON schemas that the model can use.
            tool_choice (`Literal["auto", "none", "any", "required"] | str \
            | None`, default `None`):
                Controls which (if any) tool is called by the model.
                 Can be "auto", "none", "any", "required", or specific tool
                 name. For more details, please refer to
                 https://platform.openai.com/docs/api-reference/responses/create#responses_create-tool_choice
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output. When provided, the model will be forced
                to return data that conforms to this schema by automatically
                converting the BaseModel to a tool function and setting
                `tool_choice` to enforce its usage. This enables structured
                output generation.

                .. note:: When `structured_model` is specified,
                    both `tools` and `tool_choice` parameters are ignored,
                    and the model will only perform structured output
                    generation without calling any other tools.
            callback (`Callable | None`, default `None`):
                A callback function that will be called with the response to
                process metadata. Only applicable for non-streaming responses.
            **kwargs (`Any`):
                The keyword arguments for OpenAI chat completions API,
                e.g. `temperature`, `max_tokens`, `top_p`, etc. Please
                refer to the OpenAI API documentation for more details.

        Returns:
            `ChatResponse | AsyncGenerator[ChatResponse, None]`:
                The response from the OpenAI chat completions API.
        """

        # checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, " f"got `{type(messages)}` instead.",
            )
        messages = [msg.to_dict() if isinstance(msg, ChatMessage) else msg for msg in messages]
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' and 'content' key for OpenAI API.",
            )

        # Qwen-omni requires different base64 audio format from openai
        if "omni" in self.model.lower():
            _format_audio_data_for_qwen_omni(messages)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            **self.kwargs,
            **kwargs,
        }
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        # Handle enable_thinking parameter for DashScope/Qwen models
        # For non-streaming calls with qwen models, enable_thinking must be False
        # Use extra_body to pass parameters not in OpenAI SDK
        if not self.stream and "qwen" in self.model.lower():
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            kwargs["extra_body"]["enable_thinking"] = False
            logger.debug(f"Set enable_thinking=False in extra_body for qwen model")

        if tool_choice:
            self._validate_tool_choice(tool_choice, tools)

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            kwargs.pop("stream", None)
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)

            if "qwen" in self.model:
                structured_model = {"type": "json_object"}  # type: ignore

            kwargs["response_format"] = structured_model
            if not self.stream:
                response = await self.client.chat.completions.parse(**kwargs)
            else:
                response = self.client.chat.completions.stream(**kwargs)
        else:
            response = await self.client.chat.completions.create(**kwargs)

        if self.stream:
            return self._handle_streaming_response(response, structured_model, callback)

        # Non-streaming response
        return self._handle_non_streaming_response(response, structured_model, callback)

    def _handle_non_streaming_response(
        self,
        response,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
    ) -> ChatResponse:
        """Given an OpenAI chat completion response object, extract the content
            blocks and usages from it.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (`ChatCompletion`):
                OpenAI ChatCompletion object to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            ChatResponse (`ChatResponse`):
                A ChatResponse object containing the content blocks and usage.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """

        if response.choices:
            choice = response.choices[0]
            message_data = choice.message.model_dump()
            # Ensure parsed field is present and is a dict
            message_data.setdefault("parsed", {})
            parsed_response = ChatResponse(**message_data)

            if structured_model:
                try:
                    # Check if message has parsed attribute (from chat.completions.parse)
                    if hasattr(choice.message, "parsed") and choice.message.parsed:
                        parsed = choice.message.parsed.model_dump()
                        # Ensure parsed is always a dict
                        if not isinstance(parsed, dict):
                            parsed = {"result": parsed} if parsed is not None else {}
                        parsed_response.parsed = parsed
                    else:
                        # Fallback to parsing content as JSON
                        content = choice.message.content
                        if content:
                            parsed = repair_and_load_json(content)
                            # Ensure parsed is always a dict
                            if not isinstance(parsed, dict):
                                parsed = {"result": parsed} if parsed is not None else {}
                            parsed_response.parsed = parsed
                        else:
                            parsed_response.parsed = {}
                except AttributeError:
                    content = choice.message.content
                    if content:
                        parsed = repair_and_load_json(content)
                        # Ensure parsed is always a dict
                        if not isinstance(parsed, dict):
                            parsed = {"result": parsed} if parsed is not None else {}
                        parsed_response.parsed = parsed
                    else:
                        parsed_response.parsed = {}
        else:
            raise ValueError("No choices found in the response.")

        # If callback is a function, call it with the response
        if callback and callable(callback):
            try:
                callback_result = callback(parsed_response)
                if isinstance(callback_result, dict):
                    parsed_response.parsed = parsed_response.parsed or {}
                    parsed_response.parsed.update(callback_result)
            except Exception as e:
                # Log the exception but don't fail the entire operation
                logger.warning(f"Callback function raised an exception: {type(e).__name__}: {e}", exc_info=True)

        return parsed_response

    # pylint: disable=too-many-statements
    async def _handle_streaming_response(
        self,
        response,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Handle streaming response from OpenAI API.

        Args:
            response: Async generator of ChatCompletion chunks or stream manager
            structured_model: Pydantic model for parsing structured output
            callback: Callback function to process the final response

        Yields:
            ChatResponse: Each chunk as ChatResponse object
        """
        full_content = ""
        last_chunk = None
        role = "assistant"  # Default role for responses

        # Handle different types of streaming responses
        # Some models return AsyncChatCompletionStreamManager, others return async generators
        stream = response
        if not hasattr(response, "__aiter__"):
            # If response is not directly iterable, try to get the stream
            # This handles cases like AsyncChatCompletionStreamManager
            if hasattr(response, "__stream__"):
                stream = response.__stream__()
            elif callable(getattr(response, "stream", None)):
                stream = response.stream()
            elif callable(getattr(response, "__aiter__", None)):
                stream = response
            else:
                # Last resort - try to treat it as a regular async iterable
                stream = response

        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta:
                        # Get role from first chunk if available
                        if getattr(delta, "role", None):
                            role = delta.role

                        # Accumulate content
                        if getattr(delta, "content", None):
                            full_content += delta.content

                        # Create ChatResponse from chunk with proper role
                        chunk_data = delta.model_dump()
                        chunk_data["role"] = role  # Ensure role is always set
                        chunk_response = ChatResponse(**chunk_data)
                        yield chunk_response
                        last_chunk = chunk
        except Exception as e:
            logger.warning(f"Error during streaming: {e}")
            # Even if streaming has issues, we try to yield a final response with accumulated content

        # Process final chunk with accumulated content and parsed data
        if last_chunk:
            # Create a final response with complete content
            final_choice = last_chunk.choices[0] if last_chunk.choices else None
            delta = getattr(final_choice, "delta", None) if final_choice else None
            if delta:
                # Update with full content
                delta.content = full_content
                delta.role = role

                final_response = ChatResponse(**delta.model_dump())

                # Parse structured output if needed
                if structured_model:
                    try:
                        # Try to parse the full content as structured output
                        parsed = repair_and_load_json(full_content)
                        if not isinstance(parsed, dict):
                            parsed = {"result": parsed} if parsed is not None else {}
                        final_response.parsed = parsed
                    except Exception as e:
                        logger.warning(f"Failed to parse structured output from streamed response: {e}")
                        final_response.parsed = {}

                # Apply callback if provided
                if callback and callable(callback):
                    try:
                        callback_result = callback(final_response)
                        if isinstance(callback_result, dict):
                            final_response.parsed = final_response.parsed or {}
                            final_response.parsed.update(callback_result)
                    except Exception as e:
                        logger.warning(f"Callback function raised an exception: {type(e).__name__}: {e}", exc_info=True)

                yield final_response
