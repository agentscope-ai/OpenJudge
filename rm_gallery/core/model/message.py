from datetime import datetime
from enum import Enum
from typing import Any, Generator, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Tool(BaseModel):
    """Represents a function tool with name and arguments."""

    arguments: str
    name: str


class ChatTool(BaseModel):
    """Represents a chat tool call with function metadata."""

    id: str
    function: Tool
    type: Literal["function"]


class ChatMessage(BaseModel):
    """
    Represents a chat message with role, content, and metadata.

    Attributes:
        role: Message role (system/user/assistant/function)
        name: Optional name associated with the message
        content: Main content of the message
        reasoning_content: Internal reasoning information
        tool_calls: List of tools called in this message
        additional_kwargs: Extra metadata dictionary
        time_created: Timestamp of message creation
    """

    role: MessageRole = Field(default=MessageRole.USER)
    name: Optional[str] = Field(default=None)
    content: Optional[Any] = Field(default="")
    reasoning_content: Optional[Any] = Field(default="")
    tool_calls: Optional[List[ChatTool]] = Field(default=None)
    additional_kwargs: dict = Field(default_factory=dict)
    time_created: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp marking the message creation time",
    )

    def __str__(self) -> str:
        """Returns formatted string representation with timestamp and role."""
        return f"{self.time_created.strftime('%Y-%m-%d %H:%M:%S')} {self.role.value}: {self.content}"

    def __add__(self, other: Any) -> "ChatMessage":
        """
        Concatenates message content with another message delta.

        Args:
            other: Message to merge with current one

        Returns:
            New ChatMessage instance with merged content

        Raises:
            TypeError: If other is not None or ChatMessage
        """
        if other is None:
            return self
        elif isinstance(other, ChatMessage):
            return self.__class__(
                role=self.role,
                name=self.name,
                content=self.content + (other.content if other.content else ""),
                tool_calls=other.tool_calls,
                additional_kwargs=other.additional_kwargs,
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )

    @staticmethod
    def convert_from_strings(messages: List[str], system_message: str) -> str:
        """
        Converts string list to structured ChatMessage list for debugging.

        Args:
            messages: List of alternating user/assistant messages
            system_message: Initial system message content

        Returns:
            List of structured ChatMessage objects
        """
        result_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_message),
        ]

        toggle_roles = [MessageRole.USER, MessageRole.ASSISTANT]
        for index, msg in enumerate(messages):
            result_messages.append(
                ChatMessage(role=toggle_roles[index % 2], content=msg)
            )

        return result_messages

    @staticmethod
    def convert_to_strings(messages: List["ChatMessage"]) -> Tuple[List[str], str]:
        """
        Converts structured ChatMessages to plain strings for debugging.

        Args:
            messages: List of ChatMessage objects

        Returns:
            Tuple containing:
            - List of non-system messages
            - Extracted system message content
        """
        vanilla_messages = []
        system_message = ""

        for index, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                system_message += msg.content
            else:
                vanilla_messages.append(msg.content)

        return vanilla_messages, system_message


class ChatResponse(BaseModel):
    """
    Represents a chat response with message and metadata.

    Attributes:
        message: Main chat message content
        raw: Raw response dictionary from API
        delta: Incremental update message
        error_message: Error description if any
        additional_kwargs: Extra metadata dictionary
    """

    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[ChatMessage] = None
    error_message: Optional[str] = None
    additional_kwargs: dict = Field(
        default_factory=dict
    )  # other information like token usage or log probs.

    def __str__(self):
        """Returns error message if present, otherwise string representation of main message."""
        if self.error_message:
            return f"Errors: {self.error_message}"
        else:
            return str(self.message)

    def __add__(self, other: Any) -> "ChatResponse":
        """
        Combines response with another response delta.

        Args:
            other: Response to merge with current one

        Returns:
            New ChatResponse instance with merged content

        Raises:
            TypeError: If other is not None or ChatResponse
        """
        if other is None:
            return self
        elif isinstance(other, ChatResponse):
            return self.__class__(
                message=self.message + other.message,
                raw=other.raw,
                delta=other.message,
                error_message=other.error_message,
                additional_kwargs=other.additional_kwargs,
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )


GeneratorChatResponse = Generator[ChatResponse, None, None]


def format_messages(messages: List[ChatMessage]) -> str:
    """
    Formats chat messages into XML-style string representation.

    Args:
        messages: List of ChatMessage objects to format

    Returns:
        String with messages wrapped in role-specific tags
    """
    return "\n".join(
        [f"<{message.role}>{message.content}</{message.role}>" for message in messages]
    )
