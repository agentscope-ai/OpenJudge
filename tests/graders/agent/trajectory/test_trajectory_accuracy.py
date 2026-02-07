# -*- coding: utf-8 -*-
"""
Complete demo test for TrajectoryAccuracyGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using TrajectoryAccuracyGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_accuracy.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_accuracy.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_accuracy.py -m quality -s
    ```

    Note: Use `-s` flag to see print output (evaluation summaries, etc.)
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer
from openjudge.graders.agent.trajectory.trajectory_accuracy import (
    TrajectoryAccuracyGrader,
)
from openjudge.graders.base_grader import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestTrajectoryAccuracyGraderUnit:
    """Unit tests for TrajectoryAccuracyGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = TrajectoryAccuracyGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "trajectory_accuracy"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_accurate_trajectory(self):
        """Test successful evaluation with accurate trajectory"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 3.0,  # Perfect accuracy - successfully achieves goal without unnecessary steps
            "reason": "The trajectory successfully achieves the task goal without any steps unrelated to the task",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            messages = [
                {"role": "user", "content": "What's the weather in London?"},
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": '{"temperature": 15, "condition": "cloudy"}'},
                {"role": "assistant", "content": "The weather in London is currently 15°C and cloudy."},
            ]

            result = await grader.aevaluate(messages=messages)

            # Assertions
            assert result.score == 3.0
            assert "successfully" in result.reason.lower() or "achieves" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_unnecessary_steps(self):
        """Test evaluation detecting trajectory with unnecessary steps"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 2.0,  # Successfully achieves goal but contains unnecessary steps
            "reason": "The trajectory successfully achieves the task goal but contains obvious unnecessary steps unrelated to the task",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test - trajectory with unnecessary steps
            messages = [
                {"role": "user", "content": "What's the weather in London?"},
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": '{"temperature": 15, "condition": "cloudy"}'},
                {
                    "role": "assistant",
                    "content": "Let me also check the time.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "get_time", "arguments": "{}"},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "get_time", "content": '{"time": "14:30"}'},
                {"role": "assistant", "content": "The weather in London is currently 15°C and cloudy."},
            ]

            result = await grader.aevaluate(messages=messages)

            # Assertions
            assert result.score == 2.0
            assert "unnecessary" in result.reason.lower() or "unrelated" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_failed_goal(self):
        """Test evaluation detecting trajectory that fails to achieve goal"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 1.0,  # Fails to achieve the task goal
            "reason": "The trajectory fails to achieve the task goal",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TrajectoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test - trajectory that doesn't achieve the goal
            messages = [
                {"role": "user", "content": "What's the weather in London?"},
                {
                    "role": "assistant",
                    "content": "I'll help you with that.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_time", "arguments": "{}"},
                            "type": "function",
                        }
                    ],
                },
                {"role": "tool", "name": "get_time", "content": '{"time": "14:30"}'},
                {"role": "assistant", "content": "The current time is 14:30."},
            ]

            result = await grader.aevaluate(messages=messages)

            # Assertions
            assert result.score == 1.0
            assert "fail" in result.reason.lower() or "not achieve" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test error handling when messages are empty"""
        mock_model = AsyncMock()
        grader = TrajectoryAccuracyGrader(model=mock_model)

        result = await grader.aevaluate(messages=[])

        # Should return error score
        assert result.score == 0.0
        assert "No messages" in result.reason

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when evaluation fails"""
        # Setup mock to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = TrajectoryAccuracyGrader(model=mock_model)
            grader.model.achat = mock_achat

            messages = [
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Test response"},
            ]

            result = await grader.aevaluate(messages=messages)

            # Should return error
            assert isinstance(result, GraderError)


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestTrajectoryAccuracyGraderQuality:
    """Quality tests for TrajectoryAccuracyGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Dataset with good and poor trajectory examples"""
        return [
            # Good cases - successfully achieve goal without unnecessary steps (Score 3)
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in London?"},
                    {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_weather", "content": '{"temperature": 15, "condition": "cloudy"}'},
                    {"role": "assistant", "content": "The weather in London is currently 15°C and cloudy."},
                ],
                "human_score": 3.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "Convert 100 USD to GBP"},
                    {
                        "role": "assistant",
                        "content": "I'll convert the currency for you.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "convert_currency",
                                    "arguments": '{"amount": 100, "from": "USD", "to": "GBP"}',
                                },
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "convert_currency", "content": '{"result": 78.5, "currency": "GBP"}'},
                    {"role": "assistant", "content": "100 USD is equivalent to 78.5 GBP."},
                ],
                "human_score": 3.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "Search for Python tutorials"},
                    {
                        "role": "assistant",
                        "content": "I'll search for Python tutorials.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"query": "Python tutorials"}'},
                                "type": "function",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "search",
                        "content": '[{"title": "Python Basics", "url": "https://example.com/python"}]',
                    },
                    {
                        "role": "assistant",
                        "content": "I found Python tutorials. Here's one: Python Basics at https://example.com/python",
                    },
                ],
                "human_score": 3.0,
            },
            # Medium cases - successfully achieve goal but with unnecessary steps (Score 2)
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in London?"},
                    {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_weather", "content": '{"temperature": 15, "condition": "cloudy"}'},
                    {
                        "role": "assistant",
                        "content": "Let me also check the current time.",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {"name": "get_time", "arguments": "{}"},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_time", "content": '{"time": "14:30"}'},
                    {"role": "assistant", "content": "The weather in London is currently 15°C and cloudy."},
                ],
                "human_score": 2.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "Convert 100 USD to GBP"},
                    {
                        "role": "assistant",
                        "content": "I'll convert the currency for you.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "convert_currency",
                                    "arguments": '{"amount": 100, "from": "USD", "to": "GBP"}',
                                },
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "convert_currency", "content": '{"result": 78.5, "currency": "GBP"}'},
                    {
                        "role": "assistant",
                        "content": "Let me also check the exchange rate history.",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "get_exchange_rate_history",
                                    "arguments": '{"from": "USD", "to": "GBP"}',
                                },
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_exchange_rate_history", "content": '{"history": []}'},
                    {"role": "assistant", "content": "100 USD is equivalent to 78.5 GBP."},
                ],
                "human_score": 2.0,
            },
            # Poor cases - fail to achieve goal (Score 1)
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in London?"},
                    {
                        "role": "assistant",
                        "content": "I'll help you with that.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_time", "arguments": "{}"},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_time", "content": '{"time": "14:30"}'},
                    {"role": "assistant", "content": "The current time is 14:30."},
                ],
                "human_score": 1.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "Convert 100 USD to GBP"},
                    {
                        "role": "assistant",
                        "content": "I'll help you with that.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "get_weather", "content": '{"temperature": 15, "condition": "cloudy"}'},
                    {"role": "assistant", "content": "The weather in London is 15°C and cloudy."},
                ],
                "human_score": 1.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "Search for Python tutorials"},
                    {
                        "role": "assistant",
                        "content": "I'll search for that.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "delete_file", "arguments": '{"path": "/tmp/test.py"}'},
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "name": "delete_file", "content": '{"status": "deleted"}'},
                    {"role": "assistant", "content": "File deleted successfully."},
                ],
                "human_score": 1.0,
            },
        ]

    @pytest.fixture
    def model(self):
        """Fixture to provide the model for testing"""
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
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between accurate and inaccurate trajectories"""
        # Create grader with real model
        grader = TrajectoryAccuracyGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "trajectory_accuracy": GraderConfig(
                grader=grader,
                mapper={
                    "messages": "messages",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=dataset)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["trajectory_accuracy"],
            label_path="human_score",
        )

        # Assert that accuracy metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.6, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify metadata
        assert "explanation" in accuracy_result.metadata

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test evaluation consistency across multiple runs"""
        # Create grader
        grader = TrajectoryAccuracyGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "trajectory_accuracy_run1": GraderConfig(
                grader=grader,
                mapper={
                    "messages": "messages",
                },
            ),
            "trajectory_accuracy_run2": GraderConfig(
                grader=grader,
                mapper={
                    "messages": "messages",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["trajectory_accuracy_run1"],
            another_grader_results=results["trajectory_accuracy_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.6
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"
