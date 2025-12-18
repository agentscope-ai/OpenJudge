# -*- coding: utf-8 -*-
"""The chat model base class.

This module defines the abstract base class for chat models, providing a common
interface for different LLM services such as OpenAI, DashScope, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from rm_gallery.core.models.schema.oai.response import ChatResponse

TOOL_CHOICE_MODES = ["auto", "none", "any", "required"]


class BaseChatModel(ABC):
    """Base class for chat models.

    This abstract base class defines the common interface for different chat model
    implementations. All chat model classes should inherit from this class and
    implement the required methods.

    Attributes:
        model (str): The model name.
        stream (bool): Is the model output streaming or not.

    Example:
        >>> class MyChatModel(BaseChatModel):
        ...     async def achat(self, *args, **kwargs):
        ...         # Implementation here
        ...         pass
        >>> model = MyChatModel(model="qwen3-max", stream=False)
        >>> print(model.model)
        gpt-3.5-turbo
    """

    model: str
    """The model name"""

    stream: bool
    """Is the model output streaming or not"""

    def __init__(
        self,
        model: str,
        stream: bool,
    ) -> None:
        """Initialize the chat model base class.

        Args:
            model: The name of the model.
            stream: Whether the model output is streaming or not.

        Example:
            >>> model = BaseChatModel(model="qwen3-max", stream=True)
            >>> print(model.model)
            gpt-3.5-turbo
            >>> print(model.stream)
            True
        """
        self.model = model
        self.stream = stream

    @abstractmethod
    async def achat(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Asynchronously chat with the model.

        This abstract method must be implemented by subclasses to provide
        the actual chat functionality with the specific model service.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[ChatResponse, AsyncGenerator[ChatResponse, None]]: Either a single
            ChatResponse or an async generator of ChatResponse objects for streaming.

        Example:
            >>> # This is an abstract method that must be implemented by subclasses
            >>> # Implementation would handle communication with the specific model service
        """

    def _validate_tool_choice(
        self,
        tool_choice: str,
        tools: list[dict] | None,
    ) -> None:
        """
        Validate tool_choice parameter.

        Args:
            tool_choice: Tool choice mode or function name.
            tools: Available tools list.

        Raises:
            TypeError: If tool_choice is not string.
            ValueError: If tool_choice is invalid.

        Example:
            >>> model = BaseChatModel(model="test", stream=False)
            >>> model._validate_tool_choice("auto", None)  # Valid
            >>> # model._validate_tool_choice(123, None)  # Would raise TypeError
        """
        if not isinstance(tool_choice, str):
            raise TypeError(
                f"tool_choice must be str, got {type(tool_choice)}",
            )
        if tool_choice in TOOL_CHOICE_MODES:
            return

        available_functions = [tool["function"]["name"] for tool in tools] if tools else []

        if tool_choice not in available_functions:
            all_options = TOOL_CHOICE_MODES + available_functions
            raise ValueError(
                f"Invalid tool_choice '{tool_choice}'. " f"Available options: {', '.join(sorted(all_options))}",
            )
