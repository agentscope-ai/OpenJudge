# -*- coding: utf-8 -*-
"""
Unit tests and quality tests for TrajectoryErrorRecoveryGrader.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_error_recovery.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_error_recovery.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_error_recovery.py -m quality -s
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.agent.trajectory.trajectory_error_recovery import (
    TrajectoryErrorRecoveryGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================


@pytest.mark.unit
class TestTrajectoryErrorRecoveryGraderUnit:
    """Unit tests for TrajectoryErrorRecoveryGrader - testing isolated functionality"""

    def test_initialization_default(self):
        """Test successful initialization with defaults"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "trajectory_error_recovery"
        assert grader.model == mock_model
        assert grader.language == LanguageEnum.EN

    def test_initialization_chinese(self):
        """Test initialization with Chinese language"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(
            model=mock_model,
            language=LanguageEnum.ZH,
        )
        assert grader.language == LanguageEnum.ZH

    @pytest.mark.asyncio
    async def test_good_error_recovery(self):
        """Test evaluation with good error recovery — agent adapts after error"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "The agent recognized the error, diagnosed the cause, and adapted its strategy effectively",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Search for the file config.yaml"},
                {
                    "role": "assistant",
                    "content": "I'll search for the file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": '{"path": "/wrong/dir"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "search", "content": "Error: path not found"},
                {
                    "role": "assistant",
                    "content": "The path was incorrect. Let me try the home directory.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "search", "arguments": '{"path": "/home"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "search", "content": "Found: /home/user/config.yaml"},
                {"role": "assistant", "content": "Found the file at /home/user/config.yaml"},
            ]

            result = await grader.aevaluate(messages=messages)

            assert result.score == 1.0
            assert "recognize" in result.reason.lower() or "adapt" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_poor_error_recovery(self):
        """Test evaluation with poor error recovery — agent repeats same failed action"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "The agent ignored errors and repeated the same failed action without adaptation",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Search for the file config.yaml"},
                {
                    "role": "assistant",
                    "content": "I'll search for the file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": '{"path": "/wrong/dir"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "search", "content": "Error: path not found"},
                {
                    "role": "assistant",
                    "content": "Let me try again.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "search", "arguments": '{"path": "/wrong/dir"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "search", "content": "Error: path not found"},
                {
                    "role": "assistant",
                    "content": "I couldn't find the file.",
                },
            ]

            result = await grader.aevaluate(messages=messages)

            assert result.score == 0.0
            assert (
                "repeat" in result.reason.lower()
                or "ignore" in result.reason.lower()
                or "adapt" in result.reason.lower()
            )

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling when messages are empty"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(model=mock_model)

        result = await grader.aevaluate(messages=[])

        assert result.score == 0.0
        assert "No messages" in result.reason

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when evaluation fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Test response"},
            ]

            result = await grader.aevaluate(messages=messages)

            assert result.score == 0.0
            assert "error" in result.reason.lower() or "Error" in result.reason

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test evaluation with optional context parameter"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Agent adapted strategy effectively after error",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Deploy the service"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "deploy", "arguments": '{"env": "production"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "deploy", "content": "Error: insufficient permissions"},
                {
                    "role": "assistant",
                    "content": "I need elevated permissions. Let me try with admin role.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "deploy", "arguments": '{"env": "production", "role": "admin"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "deploy", "content": "Deployed successfully"},
                {"role": "assistant", "content": "Deployment completed successfully."},
            ]

            result = await grader.aevaluate(
                messages=messages,
                context="Deploy to production environment with limited permissions",
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_score_normalization(self):
        """Test that scores are normalized to binary (0.0 or 1.0)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.7,
            "reason": "Moderate error recovery",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Test response"},
            ]

            result = await grader.aevaluate(messages=messages)

            # Score > 0.5 should be normalized to 1.0
            assert result.score == 1.0
            assert result.metadata["raw_score"] == 0.7

    @pytest.mark.asyncio
    async def test_score_normalization_below_threshold(self):
        """Test that score <= 0.5 is normalized to 0.0"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.3,
            "reason": "Poor error recovery",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Test response"},
            ]

            result = await grader.aevaluate(messages=messages)

            assert result.score == 0.0
            assert result.metadata["raw_score"] == 0.3

    def test_format_messages_string_content(self):
        """Test _format_messages with string content"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(model=mock_model)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = grader._format_messages(messages)
        assert "[user] Hello" in result
        assert "[assistant] Hi there" in result

    def test_format_messages_with_tool_calls(self):
        """Test _format_messages with tool calls"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(model=mock_model)

        messages = [
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"query": "test"}'},
                        "type": "function",
                    }
                ],
            },
        ]

        result = grader._format_messages(messages)
        assert "[assistant]" in result
        assert "Tool Calls:" in result
        assert "search" in result

    def test_format_messages_with_list_content(self):
        """Test _format_messages with multimodal list content"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(model=mock_model)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            },
        ]

        result = grader._format_messages(messages)
        assert "[user]" in result
        assert "What is in this image?" in result

    def test_format_messages_wrapped_in_message_key(self):
        """Test _format_messages with messages wrapped in 'message' key"""
        mock_model = AsyncMock()
        grader = TrajectoryErrorRecoveryGrader(model=mock_model)

        messages = [
            {"message": {"role": "user", "content": "Hello"}},
            {"message": {"role": "assistant", "content": "Hi"}},
        ]

        result = grader._format_messages(messages)
        assert "[user] Hello" in result
        assert "[assistant] Hi" in result

    @pytest.mark.asyncio
    async def test_metadata_fields(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.8,
            "reason": "Good error recovery",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryErrorRecoveryGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]

            result = await grader.aevaluate(messages=messages)

            assert "raw_score" in result.metadata
            assert "evaluation_type" in result.metadata
            assert result.metadata["evaluation_type"] == "trajectory_error_recovery"


# ==================== QUALITY TESTS ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestTrajectoryErrorRecoveryGraderQuality:
    """Quality tests for TrajectoryErrorRecoveryGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {
                "model": "qwen-plus",
                "api_key": OPENAI_API_KEY,
                "stream": False,
            }
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_good_recovery_trajectory(self, model):
        """Test quality on a trajectory with good error recovery"""
        grader = TrajectoryErrorRecoveryGrader(model=model)

        messages = [
            {"role": "user", "content": "Find the config file"},
            {
                "role": "assistant",
                "content": "I'll search in /etc directory.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"path": "/etc"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Error: Permission denied"},
            {
                "role": "assistant",
                "content": "Permission denied. Let me try the user directory instead.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "search", "arguments": '{"path": "/home/user"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Found: /home/user/config.yaml"},
            {"role": "assistant", "content": "Found the config file at /home/user/config.yaml"},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.score == 1.0
        assert result.name == "trajectory_error_recovery"

    @pytest.mark.asyncio
    async def test_poor_recovery_trajectory(self, model):
        """Test quality on a trajectory with poor error recovery"""
        grader = TrajectoryErrorRecoveryGrader(model=model)

        messages = [
            {"role": "user", "content": "Find the config file"},
            {
                "role": "assistant",
                "content": "I'll search for the file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"path": "/etc"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Error: Permission denied"},
            {
                "role": "assistant",
                "content": "Let me try again.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "search", "arguments": '{"path": "/etc"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Error: Permission denied"},
            {
                "role": "assistant",
                "content": "Trying once more.",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "function": {"name": "search", "arguments": '{"path": "/etc"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Error: Permission denied"},
            {"role": "assistant", "content": "I couldn't find the file."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.score == 0.0
