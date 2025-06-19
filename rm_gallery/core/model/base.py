import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from retry import retry

from rm_gallery.core.model.message import (
    ChatMessage,
    ChatResponse,
    GeneratorChatResponse,
    MessageRole,
)


def get_from_dict_or_env(
    data: Dict[str, Any],
    key: str,
    default: Optional[str] = None,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary or environment. This can be a list of keys to try
            in order.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
    """
    if key in data and data[key]:
        return data[key]
    elif key.upper() in os.environ and os.environ[key.upper()]:
        return os.environ[key.upper()]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{key.upper()}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )


def _convert_chat_message_to_openai_message(
    messages: List[ChatMessage],
) -> List[Dict[str, str]]:
    """Convert ChatMessage objects to OpenAI API message format dictionaries.

    Handles multiple serialization attempts to accommodate different message formats.
    """
    try:
        return [
            {
                "role": message.role.name.lower(),
                "content": message.content or "",
            }
            for message in messages
        ]
    except:
        try:
            return [
                {
                    "role": str(message.role).lower(),
                    "content": message.content or "",
                }
                for message in messages
            ]
        except:
            return [
                {
                    "role": str(message["role"]).lower(),
                    "content": str(message["content"]) or "",
                }
                for message in messages
            ]


def _convert_openai_response_to_response(response: Any) -> ChatResponse:
    """Convert OpenAI API response to ChatResponse object.

    Extracts message content and additional metadata from API response.
    """
    message = response.choices[0].message
    additional_kwargs = {"token_usage": getattr(response, "usage", {})}

    message = ChatMessage(
        role=getattr(message, "role", "assistant"),
        content=getattr(message, "content", ""),
        name=getattr(message, "name", None),
        tool_calls=getattr(message, "tool_calls", None),
        additional_kwargs=additional_kwargs,
    )

    return ChatResponse(
        message=message,
        raw=response.model_dump()
        if hasattr(response, "model_dump")
        else vars(response),
        additional_kwargs=additional_kwargs,
    )


def _convert_stream_chunk_to_response(chunk: Any) -> Optional[ChatResponse]:
    """Convert a streaming response chunk to ChatResponse object.

    Returns None if chunk contains no meaningful content.
    """
    if not chunk.choices:
        return None

    delta = chunk.choices[0].delta
    if not delta.content and not hasattr(delta, "role"):
        return None

    message = ChatMessage(
        role="assistant",
        content=delta.content or "",
        name=getattr(delta, "name", None),
        tool_calls=getattr(delta, "tool_calls", None),
        additional_kwargs={},
    )

    return ChatResponse(
        message=message,
        raw=chunk.model_dump() if hasattr(chunk, "model_dump") else vars(chunk),
        delta=message,
        additional_kwargs={"token_usage": getattr(chunk, "usage", {})},
    )


class BaseLLM(BaseModel):
    """Base class for Large Language Model implementations.

    Provides common configuration parameters and interface methods for LLMs.
    """

    model: str
    temperature: float = 0.85
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = Field(default=2048, description="Max tokens to generate for llm.")
    stop: List[str] = Field(default_factory=list, description="List of stop words")
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of tools to use"
    )
    tool_choice: Union[str, Dict] = Field(
        default="auto", description="tool choice when user passed the tool list"
    )
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay in seconds between retries"
    )
    enable_thinking: bool = Field(default=False)

    @staticmethod
    def _convert_messages(
        messages: List[ChatMessage] | ChatMessage | str,
    ) -> List[ChatMessage]:
        """Convert various input types to a list of ChatMessage objects.

        Handles string inputs, single messages, and message lists.
        """
        if isinstance(messages, list):
            return messages
        elif isinstance(messages, str):
            return [ChatMessage(content=messages, role=MessageRole.USER)]
        elif isinstance(messages, ChatMessage):
            assert messages.role == MessageRole.USER, "Only support user message."
            return [messages]
        else:
            raise ValueError(f"Invalid message type {messages}. ")

    def chat(
        self, messages: List[ChatMessage] | str, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
        """Process chat messages and generate a response.

        Args:
            messages: Input messages in various formats (list of ChatMessage, single ChatMessage, or string)
            **kwargs: Additional implementation-specific parameters

        Returns:
            ChatResponse for non-streaming responses or GeneratorChatResponse for streaming
        """
        raise NotImplementedError

    def register_tools(
        self, tools: List[Dict[str, Any]], tool_choice: Union[str, Dict]
    ):
        """Register tools for the LLM to use during response generation.

        Args:
            tools: List of tool definitions in OpenAI tool format
            tool_choice: Tool selection strategy ('auto' or specific tool definition)
        """
        self.tools = tools
        self.tool_choice = tool_choice

    def chat_batched(
        self, messages_batched: List[List[ChatMessage]] | str, **kwargs
    ) -> List[ChatResponse]:
        """Process multiple message batches concurrently.

        Args:
            messages_batched: List of message lists or single string input
            **kwargs: Same parameters as chat()

        Returns:
            List of ChatResponses in the same order as input batches
        """
        try:
            return asyncio.get_event_loop().run_until_complete(
                self._chat_batched(messages_batched, **kwargs)
            )
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop().run_until_complete(
                self._chat_batched(messages_batched, **kwargs)
            )

    async def _chat_batched(
        self, messages_batched: List[List[ChatMessage]] | str, **kwargs
    ) -> List[ChatResponse]:
        """Internal async implementation for batched chat processing.

        Should not be called directly by users.
        """
        responses = await asyncio.gather(
            *[self.achat(msg, **kwargs) for msg in messages_batched]
        )
        return responses

    async def achat(
        self, messages: List[ChatMessage] | str, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
        """Async version of chat method using thread pooling.

        Args:
            messages: Input messages in various formats
            **kwargs: Same parameters as chat()

        Returns:
            ChatResponse or GeneratorChatResponse depending on streaming configuration
        """
        result = await asyncio.to_thread(self.chat, messages, **kwargs)
        return result

    def simple_chat(
        self,
        query: str,
        history: Optional[List[str]] = None,
        sys_prompt: str = "",
        debug: bool = False,
    ) -> Any:
        """Simplified chat interface for basic query/response scenarios.

        Handles conversation history and system prompts automatically.
        """
        if self.enable_thinking:
            return self.simple_chat_reasoning(
                query=query, history=history, sys_prompt=sys_prompt, debug=debug
            )

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt)]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages += [ChatMessage(role=role, content=h)]

        # Implement retry logic with max_retries
        @retry(tries=self.max_retries, delay=self.retry_delay)
        def chat():
            response: ChatResponse = self.chat(messages)
            return response.message.content

        return chat()

    def simple_chat_reasoning(
        self,
        query: str,
        history: Optional[List[str]] = None,
        sys_prompt: str = "",
        debug: bool = False,
    ) -> Any:
        """Simplified chat interface with reasoning stream handling.

        Processes streaming responses with separate reasoning content handling.
        """
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt)]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages += [ChatMessage(role=role, content=h)]

        # Implement retry logic with max_retries
        @retry(tries=self.max_retries, delay=self.retry_delay)
        def chat():
            response: GeneratorChatResponse = self.chat(messages, stream=True)
            ans = ""
            enter_think = False
            leave_think = False
            for chunk in response:
                if chunk.delta:
                    delta = chunk.delta
                    if (
                        hasattr(delta, "reasoning_content")
                        and delta.reasoning_content is not None
                    ):
                        if not enter_think:
                            enter_think = True
                            ans += "</think>"
                        ans += delta.reasoning_content
                    elif delta.content:
                        if enter_think and not leave_think:
                            leave_think = True
                            ans += "</think>"
                        ans += delta.content

            return ans

        return chat()
