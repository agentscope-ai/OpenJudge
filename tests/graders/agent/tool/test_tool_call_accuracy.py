# -*- coding: utf-8 -*-
"""
Complete demo test for ToolCallAccuracyGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ToolCallAccuracyGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_accuracy.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_accuracy.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_accuracy.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer
from openjudge.graders.agent import ToolCallAccuracyGrader
from openjudge.graders.base_grader import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestToolCallAccuracyGraderUnit:
    """Unit tests for ToolCallAccuracyGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ToolCallAccuracyGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "tool_call_accuracy"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_accurate_call(self):
        """Test successful evaluation with accurate tool call"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 5.0,  # Perfect accuracy
            "reason": "Tool calls are fully relevant and parameters are correct",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolCallAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query=[{"role": "user", "content": "What's the weather in London?"}],
                tool_definitions=[
                    {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"location": "City name"},
                    },
                ],
                tool_calls=[
                    {
                        "name": "get_weather",
                        "arguments": {"location": "London"},
                    },
                ],
            )

            # Assertions
            assert result.score == 5.0
            assert "relevant" in result.reason.lower() or "correct" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_inaccurate_call(self):
        """Test evaluation detecting inaccurate tool call"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 1.0,  # Poor accuracy
            "reason": "Tool calls are irrelevant or parameters are incorrect",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolCallAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test - wrong tool for the task
            result = await grader.aevaluate(
                query=[{"role": "user", "content": "What's the weather in London?"}],
                tool_definitions=[
                    {"name": "get_weather", "description": "Get weather"},
                    {"name": "calculate", "description": "Perform calculations"},
                ],
                tool_calls=[
                    {
                        "name": "calculate",
                        "arguments": {"expression": "2+2"},
                    },
                ],
            )

            # Assertions
            assert result.score == 1.0
            assert "irrelevant" in result.reason.lower() or "incorrect" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when evaluation fails"""
        # Setup mock to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ToolCallAccuracyGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query=[{"role": "user", "content": "Test"}],
                tool_definitions=[{"name": "test_tool", "description": "Test"}],
                tool_calls=[{"name": "test_tool", "arguments": {}}],
            )

            # Should return error score
            assert isinstance(result, GraderError)


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestToolCallAccuracyGraderQuality:
    """Quality tests for ToolCallAccuracyGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Dataset combining tool_parameter_check and tool_selection test cases"""
        return [
            # Good cases - accurate tool calls (from parameter_check pass + selection pass)
            {
                "query": [
                    {
                        "role": "user",
                        "content": "I want to know the exchange rate from US Dollar to British Pound.",
                    },
                ],
                "tool_definitions": [
                    {
                        "name": "CurrencyConverter",
                        "description": "Converts one currency to another.",
                        "parameters": [
                            {"name": "base_currency", "type": "string", "description": "The currency to convert from."},
                            {"name": "target_currency", "type": "string", "description": "The currency to convert to."},
                            {
                                "name": "amount",
                                "type": "number",
                                "description": "The amount to convert.",
                                "optional": True,
                            },
                        ],
                    },
                ],
                "tool_calls": [
                    {
                        "name": "CurrencyConverter",
                        "arguments": {"base_currency": "USD", "target_currency": "GBP"},
                    },
                ],
                "human_score": 5.0,
            },
            {
                "query": [{"role": "user", "content": "Search for Python files modified last week"}],
                "tool_definitions": [
                    {"name": "search_files", "description": "Search for files by pattern"},
                    {"name": "git_log", "description": "Get git history"},
                ],
                "tool_calls": [
                    {"name": "search_files", "arguments": {"pattern": "*.py"}},
                    {"name": "git_log", "arguments": {"days": 7}},
                ],
                "human_score": 5.0,
            },
            {
                "query": [{"role": "user", "content": "Find restaurants near Times Square"}],
                "tool_definitions": [
                    {"name": "search_places", "description": "Search for places"},
                    {"name": "get_directions", "description": "Get directions"},
                ],
                "tool_calls": [
                    {"name": "search_places", "arguments": {"query": "restaurants", "location": "Times Square"}},
                ],
                "human_score": 5.0,
            },
            # Poor cases - inaccurate tool calls (from parameter_check fail + selection fail)
            {
                "query": [
                    {
                        "role": "user",
                        "content": "I want to know the exchange rate from US Dollar to British Pound.",
                    },
                ],
                "tool_definitions": [
                    {
                        "name": "CurrencyConverter",
                        "description": "Converts one currency to another.",
                        "parameters": [
                            {"name": "base_currency", "type": "string"},
                            {"name": "target_currency", "type": "string"},
                        ],
                    },
                ],
                "tool_calls": [
                    {
                        "name": "CurrencyConverter",
                        "arguments": {"base_currency": "Euro", "target_currency": "GBP"},  # Hallucinated "Euro"
                    },
                ],
                "human_score": 1.0,
            },
            {
                "query": [{"role": "user", "content": "Search for Python files"}],
                "tool_definitions": [
                    {"name": "search_files", "description": "Search for files"},
                    {"name": "delete_file", "description": "Delete a file"},
                ],
                "tool_calls": [
                    {"name": "delete_file", "arguments": {"path": "/tmp/test.py"}},  # Wrong tool selected
                ],
                "human_score": 1.0,
            },
            {
                "query": [{"role": "user", "content": "Get weather in Paris"}],
                "tool_definitions": [
                    {"name": "get_weather", "description": "Get weather information"},
                    {"name": "book_hotel", "description": "Book a hotel"},
                ],
                "tool_calls": [
                    {"name": "book_hotel", "arguments": {"city": "Paris"}},  # Irrelevant tool
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
        """Test the grader's ability to distinguish between accurate and inaccurate tool calls"""
        # Create grader with real model
        grader = ToolCallAccuracyGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "tool_call_accuracy": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
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
            grader_results=results["tool_call_accuracy"],
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
        grader = ToolCallAccuracyGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "tool_call_accuracy_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                },
            ),
            "tool_call_accuracy_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
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
            grader_results=results["tool_call_accuracy_run1"],
            another_grader_results=results["tool_call_accuracy_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.6
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"
