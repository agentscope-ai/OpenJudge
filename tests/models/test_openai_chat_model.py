# -*- coding: utf-8 -*-
"""Unit tests for OpenAIChatModel."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from openjudge.models.openai_chat_model import (
    OpenAIChatModel,
    _format_audio_data_for_qwen_omni,
)
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse


class PersonModelForTesting(BaseModel):
    """test model."""

    name: str
    age: int


@pytest.mark.unit
class TestOpenAIChatModel:
    """Test cases for OpenAIChatModel class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures before each test method."""
        with patch("openjudge.models.openai_chat_model.AsyncOpenAI"):
            self.model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

    def test_init(self):
        """Test initialization of OpenAIChatModel."""
        with patch("openjudge.models.openai_chat_model.AsyncOpenAI"):
            # Test basic initialization
            model = OpenAIChatModel(model="qwen3-32b")
            assert model.model == "qwen3-32b"
            assert not model.stream

            # Test initialization with custom parameters
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://custom-api.com/v1",
                stream=True,
                temperature=0.7,
            )
            assert model.model == "qwen3-32b"
            assert model.stream
            assert model.kwargs["temperature"] == 0.7

    def test_invalid_messages_type(self):
        """Test handling of invalid messages type."""
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(self.model.achat(messages="invalid"))
        assert "OpenAI `messages` field expected type `list`" in str(exc_info.value)

    def test_messages_missing_role_or_content(self):
        """Test handling of messages missing role or content."""
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(self.model.achat(messages=[{"content": "hello"}]))
        assert "Invalid message format" in str(exc_info.value)

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_achat_with_valid_messages(self, mock_async_openai):
        """Test achat method with valid messages."""
        # Mock the OpenAI API response
        mock_completion = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello! How can I help you today?",
                        role="assistant",
                    ),
                ),
            ],
            created=1234567890,
            model="qwen3-32b",
            object="chat.completion",
        )

        mock_instance = mock_async_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Create a new model instance for this test to ensure proper mocking
        with patch(
            "openjudge.models.openai_chat_model.AsyncOpenAI",
        ) as mock_openai_constructor:
            mock_openai_constructor.return_value = mock_instance
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

            # Test with valid messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]

            response = asyncio.run(model.achat(messages=messages))

            # Verify the response
            assert isinstance(response, ChatResponse)
            # Content can be either string or list depending on the message content
            assert response.content == "Hello! How can I help you today?"

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_achat_with_structured_model(self, mock_async_openai):
        """Test achat method with structured model."""
        # Mock the OpenAI API response for structured output
        mock_completion = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content='{"name": "John", "age": 30}',
                        role="assistant",
                    ),
                ),
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
        )

        mock_instance = mock_async_openai.return_value
        # For structured model, we need to mock the parse method instead of create
        mock_instance.chat.completions.parse = AsyncMock(return_value=mock_completion)

        # Create a new model instance for this test to ensure proper mocking
        with patch(
            "openjudge.models.openai_chat_model.AsyncOpenAI",
        ) as mock_openai_constructor:
            mock_openai_constructor.return_value = mock_instance
            model = OpenAIChatModel(
                model="gpt-3.5-turbo",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

            messages = [
                {
                    "role": "user",
                    "content": "Generate a person with name John and age 30",
                },
            ]

            response = asyncio.run(
                model.achat(
                    messages=messages,
                    structured_model=PersonModelForTesting,
                ),
            )

            # Verify the response
            assert isinstance(response, ChatResponse)

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_achat_with_chat_message_objects(self, mock_async_openai):
        """Test achat method with ChatMessage objects."""
        mock_completion = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello from assistant!",
                        role="assistant",
                    ),
                ),
            ],
            created=1234567890,
            model="qwen3-32b",
            object="chat.completion",
        )

        mock_instance = mock_async_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Create a new model instance for this test to ensure proper mocking
        with patch(
            "openjudge.models.openai_chat_model.AsyncOpenAI",
        ) as mock_openai_constructor:
            mock_openai_constructor.return_value = mock_instance
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

            messages = [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Hello!"),
            ]

            response = asyncio.run(model.achat(messages=messages))

            # Verify the response
            assert isinstance(response, ChatResponse)
            # Content can be either string or list depending on the message content
            assert response.content == "Hello from assistant!"

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_callback_execution(self, mock_async_openai):
        """Test callback function execution."""
        # Mock the OpenAI API response
        mock_completion = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Test response",
                        role="assistant",
                    ),
                ),
            ],
            created=1234567890,
            model="qwen3-32b",
            object="chat.completion",
        )

        mock_instance = mock_async_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Create a new model instance for this test to ensure proper mocking
        with patch(
            "openjudge.models.openai_chat_model.AsyncOpenAI",
        ) as mock_openai_constructor:
            mock_openai_constructor.return_value = mock_instance
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

        # Define a callback function
        callback_metadata = {"processed": True, "tags": ["test"]}

        def test_callback(_):
            return callback_metadata

        messages = [
            {"role": "user", "content": "Test message"},
        ]

        response = asyncio.run(model.achat(messages=messages, callback=test_callback))
        print(response)
        # Verify callback was executed and metadata was added
        assert response.parsed is not None
        assert "processed" in response.parsed
        assert response.parsed["processed"]
        assert "tags" in response.parsed
        assert response.parsed["tags"] == ["test"]

    def test_qwen_omni_audio_formatting(self):
        """Test Qwen-omni audio data formatting."""
        # Test that Qwen-omni model triggers audio formatting
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "base64encodeddata",
                            "format": "wav",
                        },
                    },
                ],
            },
        ]

        # Apply the transformation
        messages = _format_audio_data_for_qwen_omni(messages)

        # Check that the data was formatted correctly
        assert messages[0]["content"][0]["input_audio"]["data"].startswith(
            "data:;base64,",
        )


if __name__ == "__main__":
    pytest.main([__file__])
