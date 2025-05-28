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
    arguments: str
    name: str


class ChatTool(BaseModel):
    id: str
    function: Tool
    type: Literal["function"]


class ChatMessage(BaseModel):
    """Chat message."""

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
        return f"{self.time_created.strftime('%Y-%m-%d %H:%M:%S')} {self.role.value}: {self.content}"

    def __add__(self, other: Any) -> "ChatMessage":
        """
        concat message with other message delta.
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
        turn vanilla strings to structure messages for fast debugging
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
        turn structure messages to vanilla strings for fast debugging
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
    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[ChatMessage] = None
    error_message: Optional[str] = None
    additional_kwargs: dict = Field(
        default_factory=dict
    )  # other information like token usage or log probs.

    def __str__(self):
        if self.error_message:
            return f"Errors: {self.error_message}"
        else:
            return str(self.message)

    def __add__(self, other: Any) -> "ChatResponse":
        """
        concat response with other response delta.
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
