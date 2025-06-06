import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rm_gallery.core.model.message import (
    ChatMessage,
    ChatResponse,
    GeneratorChatResponse,
    MessageRole,
)
from rm_gallery.core.utils.retry import Retry


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
    retry_delay: int = Field(default=60, description="Delay in seconds between retries")
    enable_thinking: bool = Field(default=False)

    @staticmethod
    def _convert_messages(
        messages: List[ChatMessage] | ChatMessage | str,
    ) -> List[ChatMessage]:
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
        """

        Args:
            messages:
            **kwargs:

        Returns:

        """

        raise NotImplementedError

    def register_tools(
        self, tools: List[Dict[str, Any]], tool_choice: Union[str, Dict]
    ):
        self.tools = tools
        self.tool_choice = tool_choice

    def chat_batched(
        self, messages_batched: List[List[ChatMessage]] | str, **kwargs
    ) -> List[ChatResponse]:
        """

        Args:
            messages_batched: List of List of ChatMessage
            **kwargs: same with `chat`

        Returns:

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
        """
        Used by `chat_batched`, do not call this method directly.
        """
        responses = await asyncio.gather(
            *[self.achat(msg, **kwargs) for msg in messages_batched]
        )
        return responses

    async def achat(
        self, messages: List[ChatMessage] | str, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
        """

        Args:
            messages:
            **kwargs:

        Returns:

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
        def chat():
            response: ChatResponse = self.chat(messages)
            return response.message.content

        with Retry(self.max_retries) as retry:
            return retry(chat)

    def simple_chat_reasoning(
        self,
        query: str,
        history: Optional[List[str]] = None,
        sys_prompt: str = "",
        debug: bool = False,
    ) -> Any:
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
                            ans += "<think>"
                        ans += delta.reasoning_content
                    elif delta.content:
                        if enter_think and not leave_think:
                            leave_think = True
                            ans += "</think>"
                        ans += delta.content

            return ans

        with Retry(self.max_retries) as retry:
            return retry(chat)
