# -*- coding: utf-8 -*-
"""The chat model base class.

This module defines the abstract base class for chat models, providing a common
interface for different LLM services such as OpenAI, DashScope, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from openjudge.models.schema.oai.response import ChatResponse

TOOL_CHOICE_MODES = {"auto", "none", "any", "required"}


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
        >>> model = MyChatModel(model="qwen3-32b", stream=False)
        >>> print(model.model)
        qwen3-32b
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
            >>> class MyChatModel(BaseChatModel):
            ...     async def achat(self, *args, **kwargs):
            ...         pass
            >>> model = MyChatModel(model="qwen3-32b", stream=True)
            >>> print(model.model)
            qwen3-32b
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
            >>> class MyChatModel(BaseChatModel):
            ...     async def achat(self, *args, **kwargs):
            ...         pass
            >>> model = MyChatModel(model="test", stream=False)
            >>> model._validate_tool_choice("auto", None)  # Valid
        """
        if not isinstance(tool_choice, str):
            raise TypeError(
                f"tool_choice must be str, got {type(tool_choice)}",
            )

        tool_choice = tool_choice.strip()
        if not tool_choice:
            raise ValueError("`tool_choice` must be a non-empty string.")

        if tool_choice in TOOL_CHOICE_MODES:
            return

        if not tools:
            raise ValueError(
                f"Tool choice '{tool_choice}' is not a built-in mode ({', '.join(TOOL_CHOICE_MODES)}) "
                "and no tools were provided."
            )

        available_functions = set()
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise TypeError(f"Tool at index {i} is not a dictionary.")
            func = tool.get("function")
            if not isinstance(func, dict):
                raise TypeError(f"Tool at index {i} missing or invalid 'function' field.")
            name = func.get("name")
            if not isinstance(name, str):
                raise TypeError(f"Tool function name at index {i} is not a string.")
            available_functions.add(name)

        if tool_choice not in available_functions:
            all_options = sorted(TOOL_CHOICE_MODES | available_functions)
            raise ValueError(
                f"Invalid tool_choice '{tool_choice}'. " f"Available options: {', '.join(all_options)}",
            )
