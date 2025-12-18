# -*- coding: utf-8 -*-
"""Chat message schema definition.

This module defines the ChatMessage class, which represents a message in a chat conversation.
It is compatible with OpenAI Chat API format and supports both simple text messages and
rich content blocks including images, audio, video, and tool interactions.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A message in a chat conversation, compatible with OpenAI Chat API format.

    This class represents a single message in a chat conversation, supporting both
    simple text messages and rich content blocks. It includes metadata such as
    sender name, role, timestamp, and unique identifier.

    The format is aligned with OpenAI's Chat API format:
    {
        "role": "system" | "user" | "assistant" | "developer",
        "content": str | array[object],
        ...
    }

    Attributes:
        role (Literal["user", "assistant", "system"]): The role of the message sender.
        content (Union[str, List[Dict]]): The content of the message,
            either a string or a list of content part objects.
        name (str): The name of the message sender (optional in OpenAI format).
        refusal (str | None): The refusal message content if the message was refused.
        annotations (List[Dict] | None): Annotations on the message.
        audio (Dict | None): Audio content in the message.
        function_call (Dict | None): Function call information if the message contains a function call.
        tool_calls (List[Dict] | None): Tool calls if the message contains tool calls.

    Example:
        >>> # Create a simple text message
        >>> msg = ChatMessage(
        ...     role="user",
        ...     content="Hello, world!"
        ... )
        >>> print(msg.role)
        user
        >>> print(msg.content)
        Hello, world!
        >>>
        >>> # Create a message with multimodal content
        >>> msg = ChatMessage(
        ...     role="user",
        ...     content=[
        ...         {"type": "text", "text": "What's in this image?"},
        ...         {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ...     ]
        ... )
    """

    model_config: ConfigDict = ConfigDict(
        extra="allow",
    )

    role: Literal["user", "assistant", "system", "developer"] = Field(
        default="user",
        description="The role of the message sender",
    )
    content: Union[str, List[Dict[str, Any]], None] = Field(
        default="",
        description="The content of the message, either a string or a list of content part objects",
    )
    # Additional fields for OpenAI ChatCompletionMessage compatibility
    refusal: Optional[str] = Field(
        default=None,
        description="The refusal message content if the message was refused",
    )
    annotations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Annotations on the message",
    )
    audio: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Audio content in the message",
    )
    function_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Function call information if the message contains a function call",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls if the message contains tool calls",
    )

    name: str = Field(default="", description="The name of the message sender")

    def to_dict(self) -> dict:
        """Convert the message into JSON dict data compatible with OpenAI format.

        Returns:
            dict: Dictionary representation of the message in OpenAI format.

        Example:
            >>> msg = ChatMessage(role="user", content="Hello")
            >>> data = msg.to_dict()
            >>> print("role" in data and "content" in data)
            True
        """
        result = {
            "role": self.role,
            "content": self.content,
        }

        # Add optional fields if they exist
        if self.refusal is not None:
            result["refusal"] = self.refusal

        if self.annotations is not None:
            result["annotations"] = self.annotations

        if self.audio is not None:
            result["audio"] = self.audio

        if self.function_call is not None:
            result["function_call"] = self.function_call

        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls

        # Add name if it's not empty
        if self.name:
            result["name"] = self.name

        return result

    @classmethod
    def from_dict(cls, json_data: dict) -> "ChatMessage":
        """Load a message object from the given JSON data.

        Args:
            json_data: JSON data to load from.

        Returns:
            ChatMessage: Instance created from the JSON data.

        Example:
            >>> data = {"role": "user", "content": "Hello"}
            >>> msg = ChatMessage.from_dict(data)
            >>> print(msg.role)
            user
            >>> print(msg.content)
            Hello
        """
        return cls(**json_data)

    def get_text_content(self) -> str | None:
        """Get the pure text content from the message.

        Returns:
            str | None: The text content, or None if no text content exists.

        Example:
            >>> msg = ChatMessage(content="Hello world")
            >>> print(msg.get_text_content())
            Hello world
            >>>
            >>> msg = ChatMessage(content=[{"type": "text", "text": "Hello world"}])
            >>> print(msg.get_text_content())
            Hello world
        """
        if isinstance(self.content, str):
            return self.content

        if isinstance(self.content, list):
            # Extract text from multimodal content
            text_parts = []
            for part in self.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                return "".join(text_parts)

        return None

    def format(self, **kwargs) -> "ChatMessage":
        """Format the message content with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the message content with.

        Returns:
            ChatMessage: A copy of the message with formatted content.

        Example:
            >>> msg = ChatMessage(content="Hello {name}!")
            >>> formatted_msg = msg.format(name="Alice")
            >>> print(formatted_msg.get_text_content())
            Hello Alice!
        """
        message = self.model_copy()
        if isinstance(message.content, str):
            message.content = message.content.format(**kwargs)
        elif isinstance(message.content, list):
            formatted_content = []
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_part = part.copy()
                    text_part["text"] = text_part["text"].format(**kwargs)
                    formatted_content.append(text_part)
                else:
                    formatted_content.append(part)
            message.content = formatted_content
        return message
