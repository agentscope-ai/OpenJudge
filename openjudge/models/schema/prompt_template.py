# -*- coding: utf-8 -*-
"""Template system for chat message generation.

This module provides a flexible template system for generating chat messages
in various formats and languages. It supports both monolingual and multilingual
templates with easy formatting capabilities.
"""

from enum import Enum
from typing import Any, Dict, List, TypedDict, Union

from pydantic import BaseModel, Field

from openjudge.models.schema.oai.message import ChatMessage


class LanguageEnum(str, Enum):
    """Language enumeration for templates.

    This enum defines the supported languages for multilingual templates.
    Currently supports English (EN) and Chinese (ZH).

    Attributes:
        EN: English language code.
        ZH: Chinese language code.
        ANY: Any language code.

    Example:
        >>> print(LanguageEnum.EN)
        en
        >>> print(LanguageEnum.ZH)
        zh
        >>> print(LanguageEnum.ANY)
        any
    """

    EN = "en"
    ZH = "zh"
    ANY = "any"


class PromptDict(TypedDict, total=False):
    """Typed dictionary for structured prompts.

    A typed dictionary that defines the structure of a prompt with optional
    system and user messages. Either value can be either a string or a
    ChatMessage object.

    Attributes:
        system: Optional system message as string or ChatMessage.
        user: Optional user message as string or ChatMessage.

    Example:
        >>> prompt_dict: PromptDict = {"system": "You are a helpful assistant",
        ...                             "user": "Hello!"}
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


class PromptTemplate(BaseModel):
    """Template for generating chat messages.

    A template class that holds messages for chat generation. It supports both
    monolingual and multilingual templates. For multilingual templates, messages
    are stored in a dictionary with LanguageEnum keys.

    Attributes:
        messages: Either a list of ChatMessage objects for monolingual templates
            or a dictionary mapping LanguageEnum to lists of ChatMessage objects
            for multilingual templates.

    Example:
        >>> # Monolingual template
        >>> template = PromptTemplate(messages=[ChatMessage(role="user", content="Hello")])
        >>>
        >>> # Multilingual template
        >>> multi_template = PromptTemplate(
        ...     messages={
        ...         LanguageEnum.EN: [ChatMessage(role="user", content="Hello")],
        ...         LanguageEnum.ZH: [ChatMessage(role="user", content="你好")]
        ...     }
        ... )
    """

    messages: (
        List[ChatMessage]
        | Dict[
            LanguageEnum,
            List[ChatMessage],
        ]
    ) = Field(
        default_factory=list,
        description="messages for generating chat",
    )

    def to_messages(
        self,
        language: LanguageEnum | None = LanguageEnum.EN,
    ) -> List[ChatMessage]:
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
            >>> template = PromptTemplate(messages=[ChatMessage(role="user", content="Hello")])
            >>> template.to_messages()
            [ChatMessage(role="user", content="Hello")]

            >>> messages = {LanguageEnum.EN: [ChatMessage(role="user", content="Hello")]}
            >>> template = PromptTemplate(messages=messages)
            >>> template.to_messages(LanguageEnum.EN)
            [ChatMessage(role="user", content="Hello")]
        """
        if isinstance(self.messages, list):
            messages = self.messages
        elif isinstance(self.messages, dict):
            if not language:
                language = LanguageEnum.EN
            assert language in self.messages
            messages = self.messages.get(language, [])
        else:
            raise ValueError("Invalid messages")

        return messages

    @classmethod
    def from_prompt(cls, prompt: Prompt) -> "PromptTemplate":
        """Create a PromptTemplate instance from a prompt.

        This method converts a prompt in various formats to a PromptTemplate instance
        with a monolingual message list.

        Args:
            prompt: The prompt in various formats:
                - str: A simple string user message
                - List[ChatMessage]: Already formatted messages
                - PromptDict: Dictionary with system/user keys

        Returns:
            PromptTemplate: A new PromptTemplate instance with messages converted from the prompt.

        Examples:
            >>> template = PromptTemplate.from_prompt("Hello")
            >>> template.to_messages()
            [ChatMessage(role='user', content='Hello')]

            >>> messages = [ChatMessage(role='system', content='You are a bot')]
            >>> template = PromptTemplate.from_prompt(messages=messages)
            >>> template.to_messages()
            [ChatMessage(role='system', content='You are a bot')]
        """
        messages = _convert_prompt_to_messages(prompt)
        return cls(messages=messages)

    @classmethod
    def from_multilingual(
        cls,
        prompt: Dict[LanguageEnum | str, Prompt],
    ) -> "PromptTemplate":
        """Create a PromptTemplate instance from a multilingual prompt.

        This method creates a PromptTemplate with multilingual support by converting
        prompts for different languages into the appropriate format.

        Args:
            prompt: A dictionary mapping language codes (LanguageEnum or string)
                to prompts in various formats (str, List[ChatMessage], or PromptDict).

        Returns:
            PromptTemplate: A new PromptTemplate instance with multilingual messages.

        Examples:
            >>> prompt_dict = {
            ...     LanguageEnum.EN: "Hello",
            ...     LanguageEnum.ZH: "你好"
            ... }
            >>> template = PromptTemplate.from_multilingual(prompt_dict)
            >>> len(template.messages)
            2
        """
        return cls(
            messages={LanguageEnum(lang): _convert_prompt_to_messages(prompt) for lang, prompt in prompt.items()},
        )

    def format(
        self,
        language: LanguageEnum | None = None,
        **kwargs: Any,
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
            >>> template = PromptTemplate(messages=[ChatMessage(role="user", content="Hello {name}")])
            >>> template.format(name="World")
            [{'role': 'user', 'content': 'Hello World'}]
        """
        messages = self.to_messages(language)
        messages = [ChatMessage(**message) if not isinstance(message, ChatMessage) else message for message in messages]
        messages = [message.format(**kwargs).to_dict() for message in messages]
        return messages

    def get_prompt(self, language: LanguageEnum = None) -> Dict[str, List[Dict[str, str]]]:
        """Return the core prompts (role, content) information of the messages,
        in a {language: list[{'role': txt, 'content': txt}]} dictionary.
        """
        # Only one list of messages without language label
        if isinstance(self.messages, list):
            prompt = []
            for msg in self.messages:
                part = {"role": msg.role, "content": msg.content}
                prompt.append(part)
            return {LanguageEnum.ANY.value: prompt}

        # return data of specified language, or all data if no language restriction
        if isinstance(self.messages, dict):
            language_prompts = {}
            for lang, msgs in self.messages.items():
                if language and language != LanguageEnum.ANY and language != lang:
                    continue
                prompt = []
                for msg in msgs:
                    part = {"role": msg.role, "content": msg.content}
                    prompt.append(part)
                language_prompts[lang.value] = prompt
            return language_prompts

        raise ValueError("Cannot get prompts because of invalid messages")
