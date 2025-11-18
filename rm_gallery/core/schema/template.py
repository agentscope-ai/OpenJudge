# -*- coding: utf-8 -*-
import asyncio
import json
from abc import ABC
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Type,
    TypedDict,
    Union,
)

import yaml
from pydantic import BaseModel, Field

from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.response import ChatResponse
from rm_gallery.core.utils.instance import init_instance_by_config


class LanguageEnum(str, Enum):
    """Language enumeration for templates.

    This enum defines the supported languages for multilingual templates.
    Currently supports English (EN) and Chinese (ZH).

    Attributes:
        EN: English language code.
        ZH: Chinese language code.
    """

    EN = "en"
    ZH = "zh"


class PromptDict(TypedDict, total=False):
    """Typed dictionary for structured prompts.

    A typed dictionary that defines the structure of a prompt with optional
    system and user messages. Either value can be either a string or a
    ChatMessage object.

    Attributes:
        system: Optional system message as string or ChatMessage.
        user: Optional user message as string or ChatMessage.
    """

    system: str | ChatMessage
    user: str | ChatMessage


Prompt = Union[str, PromptDict, List[ChatMessage]]


def _convert_prompt_to_messages(prompt: Prompt) -> List[ChatMessage]:
    """Convert various prompt formats to List[ChatMessage].

    This function takes a prompt in various formats (string, list of ChatMessage
    objects, or PromptDict) and converts it to a standardized list of
    ChatMessage objects.

    Args:
        prompt: The prompt in various formats:
            - str: A simple string user message
            - List[ChatMessage]: Already formatted messages
            - PromptDict: Dictionary with system/user keys

    Returns:
        List[ChatMessage]: A list of ChatMessage objects.

    Raises:
        ValueError: If the prompt type is not supported.

    Examples:
        >>> _convert_prompt_to_messages("Hello")
        [ChatMessage(role='user', content='Hello')]

        >>> _convert_prompt_to_messages([ChatMessage(role='system', content='You are a bot')])
        [ChatMessage(role='system', content='You are a bot')]
    """
    if isinstance(prompt, str):
        return [ChatMessage(role="user", content=prompt)]
    elif isinstance(prompt, list):
        # Already in the correct format
        return prompt
    elif isinstance(prompt, dict):
        messages = []
        if "system" in prompt:
            system_content = prompt["system"]
            if isinstance(system_content, str):
                messages.append(
                    ChatMessage(role="system", content=system_content),
                )
            else:
                messages.append(system_content)
        if "user" in prompt:
            user_content = prompt["user"]
            if isinstance(user_content, str):
                messages.append(ChatMessage(role="user", content=user_content))
            else:
                messages.append(user_content)
        return messages
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")


class Template(BaseModel):
    """Template for generating chat messages.

    A template class that holds messages for chat generation. It supports both
    monolingual and multilingual templates. For multilingual templates, messages
    are stored in a dictionary with LanguageEnum keys.

    Attributes:
        messages: Either a list of ChatMessage objects for monolingual templates
            or a dictionary mapping LanguageEnum to lists of ChatMessage objects
            for multilingual templates.
    """

    messages: List[ChatMessage] | Dict[
        LanguageEnum,
        List[ChatMessage],
    ] = Field(
        default_factory=list,
        description="messages for generating chat",
    )

    def to_messages(self, language: LanguageEnum | None = LanguageEnum.EN):
        """Extract messages for the specified language.

        For monolingual templates, returns the messages directly.
        For multilingual templates, returns messages for the specified language.

        Args:
            language: The language to extract messages for. Defaults to English.
                If None, defaults to English. Only used for multilingual templates.

        Returns:
            List[ChatMessage]: The messages for the specified language.

        Raises:
            AssertionError: If the specified language is not available in a
                multilingual template.
            ValueError: If messages format is invalid.

        Examples:
            >>> template = Template(messages=[ChatMessage(role="user", content="Hello")])
            >>> template.to_messages()
            [ChatMessage(role="user", content="Hello")]

            >>> template = Template(messages={LanguageEnum.EN: [ChatMessage(role="user", content="Hello")]})
            >>> template.to_messages(LanguageEnum.EN)
            [ChatMessage(role="user", content="Hello")]
        """
        if isinstance(self.messages, list):
            messages = self.messages
        elif isinstance(self.messages, dict):
            if language is None:
                language = LanguageEnum.EN
            assert language in self.messages
            messages = self.messages.get(language, [])
        else:
            raise ValueError("Invalid messages")

        return messages

    @classmethod
    def from_prompt(cls, prompt: Prompt) -> "Template":
        """Create a Template instance from a prompt.

        This method converts a prompt in various formats to a Template instance
        with a monolingual message list.

        Args:
            prompt: The prompt in various formats:
                - str: A simple string user message
                - List[ChatMessage]: Already formatted messages
                - PromptDict: Dictionary with system/user keys

        Returns:
            Template: A new Template instance with messages converted from the prompt.

        Examples:
            >>> template = Template.from_prompt("Hello")
            >>> template.to_messages()
            [ChatMessage(role='user', content='Hello')]

            >>> template = Template.from_prompt([ChatMessage(role='system', content='You are a bot')])
            >>> template.to_messages()
            [ChatMessage(role='system', content='You are a bot')]
        """
        messages = _convert_prompt_to_messages(prompt)
        return cls(messages=messages)

    @classmethod
    def from_multilingual(
        cls,
        prompt: Dict[LanguageEnum | str, Prompt],
    ) -> "Template":
        """Create a Template instance from a multilingual prompt.

        This method creates a Template with multilingual support by converting
        prompts for different languages into the appropriate format.

        Args:
            prompt: A dictionary mapping language codes (LanguageEnum or string)
                to prompts in various formats (str, List[ChatMessage], or PromptDict).

        Returns:
            Template: A new Template instance with multilingual messages.

        Examples:
            >>> prompt_dict = {
            ...     LanguageEnum.EN: "Hello",
            ...     LanguageEnum.ZH: "你好"
            ... }
            >>> template = Template.from_multilingual(prompt_dict)
            >>> len(template.messages)
            2
        """
        return cls(
            messages={
                LanguageEnum(lang): _convert_prompt_to_messages(prompt)
                for lang, prompt in prompt.items()
            },
        )


class Chat(ABC):
    """Chat for generating response.

    A chat class that uses a template and a chat model to generate responses.
    It handles message formatting, model calling, and response processing.

    Attributes:
        template: The Template instance containing chat messages.
        model: The chat model used for generating responses.
    """

    def __init__(self, template: Template | dict, model: dict | ChatModelBase):
        """Initialize a Chat instance.

        Args:
            template: Either a Template instance or a dictionary that can be
                used to create a Template.
            model: Either a ChatModelBase instance or a dictionary configuration
                that can be used to create a chat model.
        """
        self.template = (
            template
            if isinstance(template, Template)
            else Template(**template)
        )
        self.model = init_instance_by_config(model, accept_type=ChatModelBase)

    def format(
        self,
        language: LanguageEnum | None = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Format messages with provided keyword arguments.

        This method formats the template messages by replacing placeholders with
        provided values. It returns a list of message dictionaries ready to be
        sent to a chat model.

        Args:
            language: Language code for multilingual templates. Defaults to None.
            **kwargs: Keyword arguments to format message content with.

        Returns:
            List[Dict[str, Any]]: A list of formatted message dictionaries.

        Examples:
            >>> template = Template(messages=[ChatMessage(role="user", content="Hello {name}")])
            >>> chat = Chat(template=template, model={"model_name": "qwen-plus"})
            >>> chat.format(name="World")
            [{'role': 'user', 'content': 'Hello World'}]
        """
        messages = self.template.to_messages(language)
        messages = [message.to_dict() for message in messages]

        for message in messages:
            message["content"] = message.get("content", "").format(**kwargs)
        return messages

    async def __call__(
        self,
        callback: Type[BaseModel] | Callable | None = None,
        language: LanguageEnum | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate chat response using the template.

        This method formats messages using the provided arguments, calls the chat
        model, and processes the response. It supports structured output and
        callback functions.

        Args:
            callback: Optional callback mechanism:
                - If a Pydantic BaseModel subclass, used for structured output
                - If a callable, called with the response to process metadata
                - If None, no special processing
            language: Language code for multilingual templates. Defaults to None.
            **kwargs: Keyword arguments for formatting message content.

        Returns:
            ChatResponse: The chat response from the model.

        Examples:
            >>> template = Template(messages=[ChatMessage(role="user", content="Say hello")])
            >>> chat = Chat(template=template, model={"model_name": "qwen-plus"})
            >>> response = asyncio.run(chat())
            >>> isinstance(response, ChatResponse)
            True
        """
        messages = self.format(language=language, **kwargs)
        # Check if callback is a Pydantic BaseModel class
        if (
            callback
            and isinstance(callback, type)
            and issubclass(callback, BaseModel)
        ):
            # If callback is a Pydantic class, pass it as structured_model
            response = await self.model(
                messages=messages,
                structured_model=callback,
            )
        else:
            # If callback is not a Pydantic class or is None, don't pass structured_model
            response = await self.model(
                messages=messages,
            )

        # Handle case where response might be an AsyncGenerator
        if isinstance(response, AsyncGenerator):
            # For streaming responses, collect all chunks
            content_parts = []
            metadata = {}
            usage = None

            async for chunk in response:
                content_parts.extend(chunk.content)
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                if chunk.usage:
                    usage = chunk.usage

            # Create a consolidated response
            response = ChatResponse(
                content=content_parts,
                metadata=metadata or None,
                usage=usage,
            )

        # If callback is a function, call it with the response
        if (
            callback
            and not isinstance(callback, type)
            and isinstance(callback, Callable)
        ):
            response.metadata = response.metadata or {}
            response.metadata.update(callback(response))

        return response

    @classmethod
    def load(cls, path: str):
        """Load a Chat instance from a JSON or YAML file.

        This method loads a Chat configuration from a file and creates a Chat
        instance. Supported formats are JSON (.json) and YAML (.yaml/.yml).

        Args:
            path: Path to the configuration file.

        Returns:
            Chat: A new Chat instance loaded from the file.

        Raises:
            ValueError: If the file format is not supported.

        Examples:
            >>> # Assuming a valid chat_config.json file exists
            >>> # chat = Chat.load("chat_config.json")
        """
        if path.endswith("json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif path.endswith("yaml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Invalid file format")
        return cls(**data)


if __name__ == "__main__":
    template = Template(
        messages={
            LanguageEnum.EN: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "{question}"},
            ],
        },
    )
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chat = Chat(template=template, model=model)
    messages = chat.format(
        language="en",
        question="What is the capital of France?",
    )
    print(messages)
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
