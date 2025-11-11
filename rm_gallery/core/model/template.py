import asyncio
import json
import re
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Type

import yaml
from pydantic import BaseModel, Field, model_validator

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.model.response import ChatResponse
from rm_gallery.core.model.utils import _json_loads_with_repair


class LanguageEnum(str, Enum):
    """Language enumeration for templates."""

    EN = "en"
    ZH = "zh"


class RequiredField(BaseModel):
    """Required fields for grading."""

    name: str = Field(default=..., description="name of the field")
    type: str = Field(default=..., description="type of the field")
    position: Literal["data", "sample", "grader"] = Field(
        default="data", description="position of the field"
    )
    description: str = Field(default=..., description="description of the field")


class Template(BaseModel):
    """Template for generating chat messages."""

    messages: List[ChatMessage] | Dict[LanguageEnum, List[ChatMessage]] = Field(
        default_factory=list, description="messages for generating chat"
    )
    required_fields: List[RequiredField] = Field(
        default_factory=list, description="required kwargs"
    )

    @model_validator(mode="before")
    def validate_template(cls, values) -> dict:
        messages = values.get("messages", [])
        # Pattern to match placeholders like {word}, {word.word}, {word.word.word}, etc.
        placeholder_pattern = (
            r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"
        )

        # Handle both dict and RequiredField instances
        required_fields_raw = values.get("required_fields", [])
        required_fields = []
        for field in required_fields_raw:
            if isinstance(field, dict):
                required_fields.append(field.get("name"))
            elif isinstance(field, RequiredField):
                required_fields.append(field.name)
            elif hasattr(field, "name"):
                required_fields.append(field.name)
        required = []

        all_messages = []
        if isinstance(messages, list):
            messages = messages
        elif isinstance(messages, dict):
            for language, language_messages in messages.items():
                if not isinstance(language_messages, list):
                    raise ValueError("Invalid message format")
                all_messages.extend(language_messages)
        else:
            raise ValueError("Invalid message format")

        for message in all_messages:
            content = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            # Find all placeholders in the content
            placeholders = re.findall(placeholder_pattern, content)
            for placeholder in placeholders:
                # Add to required if not already present
                if placeholder not in required:
                    required.append(placeholder)

        for name in required:
            if name not in required_fields:
                raise ValueError(f"Required field {name} is missing")
        return values

    def get(self, language: LanguageEnum = LanguageEnum.EN):
        if isinstance(self.messages, list):
            messages = self.messages
        elif isinstance(self.messages, dict):
            assert language in self.messages
            messages = self.messages.get(language, [])
        else:
            raise ValueError("Invalid messages")

        return messages


class ChatTemplate(BaseModel):
    """Chat template for generating chat messages."""

    template: Template = Field(
        default=..., description="template for generating chat messages"
    )
    model: Dict[str, Any] = Field(default=..., description="model parameters")

    @property
    def required_fields(
        self,
    ) -> List[RequiredField]:
        """Get required fields for the template."""
        return self.template.required_fields

    def format(
        self, language: LanguageEnum = LanguageEnum.EN, **kwargs
    ) -> List[Dict[str, Any]]:
        """Format messages with provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format with

        Returns:
            List of formatted message dictionaries
        """
        messages = self.template.get(language)
        messages = [message.to_dict() for message in messages]

        for message in messages:
            message["content"] = message.get("content", "").format(**kwargs)
        return messages

    async def __call__(
        self, chat_output: Callable | Type[BaseModel] | None = None, **kwargs
    ) -> ChatResponse:
        """Generate chat response using the template.

        Args:
            chat_output: Optional chat output type or callable
            **kwargs: Keyword arguments for formatting messages

        Returns:
            Chat response
        """
        messages = self.format(**kwargs)

        params = {}
        if isinstance(chat_output, type) and issubclass(chat_output, BaseModel):
            if "qwen" in self.model.get("model_name", ""):
                params["response_format"] = {"type": "json_object"}
            else:
                params["structured_model"] = chat_output

        response = await OpenAIChatModel(**self.model)(messages=messages, **params)

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
                content=content_parts, metadata=metadata or None, usage=usage
            )

        if chat_output is not None:
            metadata = response.metadata if response.metadata else {}
            text_content = ""

            # Extract text from content blocks
            for content_block in response.content:
                if getattr(content_block, "type", "") == "text":
                    text_content = getattr(content_block, "text", "{}")
                    break

            if isinstance(chat_output, type) and issubclass(chat_output, BaseModel):
                try:
                    parsed_data = _json_loads_with_repair(text_content)
                    if isinstance(parsed_data, dict):
                        metadata.update(parsed_data)
                except Exception:
                    # If parsing fails, leave metadata as is
                    pass
            elif isinstance(chat_output, Callable):
                try:
                    parsed_data = chat_output(text_content)
                    if isinstance(parsed_data, dict):
                        metadata.update(parsed_data)
                except Exception:
                    # If parsing fails, leave metadata as is
                    pass

            response.metadata = metadata

        return response

    @classmethod
    def load(cls, path: str):
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
    chat = ChatTemplate(
        template=[
            ChatMessage(
                role="system", content="You are a helpful assistant.", name="System"
            ),
            ChatMessage(role="user", content="{question}", name="User"),
        ],
        model={
            "model_name": "qwen-plus",
            "api_key": "your-real-api-key",
            "stream": False,
            "client_args": {
                "timeout": 60,
                "base_url": "https://api.openai.com/v1",
            },
        },
    )
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
