# -*- coding: utf-8 -*-
"""Unit tests for OpenAIChatModel."""

import asyncio
import base64
import copy
import io
import wave
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

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_qwen_enable_thinking_user_value_preserved(self, mock_async_openai):
        """Test that user-provided enable_thinking in extra_body is not overwritten for Qwen models."""
        mock_instance = mock_async_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=ChatCompletion(
                id="test-id",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(content="OK", role="assistant"),
                    ),
                ],
                created=1234567890,
                model="qwen3-32b",
                object="chat.completion",
            )
        )

        with patch("openjudge.models.openai_chat_model.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = mock_instance
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                extra_body={"enable_thinking": True},
            )

            messages = [{"role": "user", "content": "Hello"}]
            asyncio.run(model.achat(messages=messages))

            call_kwargs = mock_instance.chat.completions.create.call_args[1]
            # User's value should be preserved
            assert call_kwargs["extra_body"]["enable_thinking"] is True

    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_qwen_enable_thinking_default_when_not_provided(self, mock_async_openai):
        """Test that enable_thinking defaults to False for Qwen models when not provided."""
        mock_instance = mock_async_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=ChatCompletion(
                id="test-id",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(content="OK", role="assistant"),
                    ),
                ],
                created=1234567890,
                model="qwen3-32b",
                object="chat.completion",
            )
        )

        with patch("openjudge.models.openai_chat_model.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = mock_instance
            model = OpenAIChatModel(
                model="qwen3-32b",
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )

            messages = [{"role": "user", "content": "Hello"}]
            asyncio.run(model.achat(messages=messages))

            call_kwargs = mock_instance.chat.completions.create.call_args[1]
            # Default should be False
            assert call_kwargs["extra_body"]["enable_thinking"] is False

    @pytest.mark.parametrize(
        "init_kwargs, expected_in_call, not_expected_in_call",
        [
            (
                {"max_retries": 5, "timeout": 120.0},
                {"max_retries": 5, "timeout": 120.0},
                [],
            ),
            ({}, {}, ["max_retries", "timeout"]),
            ({"max_retries": None, "timeout": None}, {}, ["max_retries", "timeout"]),
            ({"max_retries": 3}, {"max_retries": 3}, ["timeout"]),
            ({"timeout": 30.0}, {"timeout": 30.0}, ["max_retries"]),
        ],
        ids=[
            "with_retries_and_timeout",
            "defaults",
            "with_none_values",
            "with_retries_only",
            "with_timeout_only",
        ],
    )
    @patch("openjudge.models.openai_chat_model.AsyncOpenAI")
    def test_client_initialization_with_retries_and_timeout(
        self,
        mock_async_openai,
        init_kwargs,
        expected_in_call,
        not_expected_in_call,
    ):
        """Test that max_retries and timeout parameters are passed to AsyncOpenAI client correctly."""
        OpenAIChatModel(
            model="gpt-4",
            api_key="test-key",
            **init_kwargs,
        )

        mock_async_openai.assert_called_once()
        call_kwargs = mock_async_openai.call_args[1]

        assert call_kwargs["api_key"] == "test-key"

        for key, value in expected_in_call.items():
            assert key in call_kwargs
            assert call_kwargs[key] == value

        for key in not_expected_in_call:
            assert key not in call_kwargs

    @staticmethod
    def _mock_completion() -> ChatCompletion:
        """Build a minimal completion returned by the mocked API."""
        return ChatCompletion(
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
            model="test-model",
            object="chat.completion",
        )

    def _sent_response_format(self, model_name: str) -> object:
        """Run one structured-output call and return the response_format sent."""
        with patch("openjudge.models.openai_chat_model.AsyncOpenAI") as mock_constructor:
            mock_instance = mock_constructor.return_value
            mock_instance.chat.completions.parse = AsyncMock(return_value=self._mock_completion())

            model = OpenAIChatModel(
                model=model_name,
                api_key="test-key",
                base_url="https://example.com/v1",
            )
            asyncio.run(
                model.achat(
                    messages=[{"role": "user", "content": "hi"}],
                    structured_model=PersonModelForTesting,
                ),
            )
            return mock_instance.chat.completions.parse.call_args[1]["response_format"]

    def test_deepseek_falls_back_to_json_object(self):
        """deepseek rejects a JSON schema, so it must receive a plain json_object."""
        sent = self._sent_response_format("deepseek-chat")

        assert sent == {"type": "json_object"}

    def test_non_deepseek_keeps_json_schema(self):
        """Models that support strict schemas must keep receiving them."""
        sent = self._sent_response_format("gpt-4o")

        assert sent is PersonModelForTesting


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("message_kind", ["dict", "chat_message"])
@pytest.mark.parametrize("data_kind", ["raw", "prefixed", "url"])
async def test_omni_audio_is_formatted_once_without_changing_history(message_kind: str, data_kind: str) -> None:
    """Repeated public calls preserve caller data and send one audio prefix."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 16)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    formatted = "data:;base64," + encoded
    source = {"raw": encoded, "prefixed": formatted, "url": "https://example.com/audio.wav"}[data_kind]
    message = {
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": source, "format": "wav"}}],
    }
    messages = [ChatMessage(**message) if message_kind == "chat_message" else message]
    original = copy.deepcopy(messages)
    completion = ChatCompletion(
        id="test-audio",
        object="chat.completion",
        created=1,
        model="qwen3-omni-flash",
        choices=[
            Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="heard"))
        ],
    )
    with patch("openjudge.models.openai_chat_model.AsyncOpenAI") as client_factory:
        create = AsyncMock(return_value=completion)
        client_factory.return_value.chat.completions.create = create
        model = OpenAIChatModel(model="qwen3-omni-flash", api_key="test-key")
        for _ in range(2):
            result = await model.achat(messages=messages)
            assert result.content == "heard"
            sent = create.call_args.kwargs["messages"][0]["content"][0]["input_audio"]
            assert sent == {"data": source if data_kind == "url" else formatted, "format": "wav"}
            assert messages == original


if __name__ == "__main__":
    pytest.main([__file__])
