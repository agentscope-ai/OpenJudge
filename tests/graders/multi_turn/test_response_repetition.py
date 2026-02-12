#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ResponseRepetitionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ResponseRepetitionGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_response_repetition.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_response_repetition.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_response_repetition.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.response_repetition_grader import (
    ResponseRepetitionGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestResponseRepetitionGraderUnit:
    """Unit tests for ResponseRepetitionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ResponseRepetitionGrader(model=mock_model)
        assert grader.name == "response_repetition"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_no_repetition(self):
        """Test evaluation when response has no repetition"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Response is completely original with no repetition.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseRepetitionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data."},
                {"role": "user", "content": "What about tomorrow?"},
            ]
            response = "For weather forecasts, I recommend checking weather.com or your local news."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "response_repetition"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_with_repetition(self):
        """Test evaluation when response repeats previous content"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Response largely repeats the same information.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseRepetitionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data."},
                {"role": "user", "content": "What about tomorrow?"},
            ]
            response = "I don't have access to real-time weather data."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "response_repetition"
            assert result.score == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ResponseRepetitionGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [{"role": "user", "content": "Hello"}]
            result = await grader.aevaluate(response="Hi!", history=history)

            # Assertions - error is returned as GraderError with error field
            assert "Evaluation error: API Error" in result.error


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

pytestmark = pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)


@pytest.mark.quality
class TestResponseRepetitionGraderQuality:
    """Quality tests for ResponseRepetitionGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with repetitive and non-repetitive examples"""
        return [
            # Case 1: No repetition - provides new information
            {
                "history": [
                    {"role": "user", "content": "Tell me about Python."},
                    {
                        "role": "assistant",
                        "content": "Python is a high-level programming language known for its simplicity.",
                    },
                    {"role": "user", "content": "What about its use cases?"},
                ],
                "response": "Python is widely used in web development, data science, AI, and automation.",
                "human_score": 1,  # No repetition (good)
            },
            # Case 2: Exact repetition - same response
            {
                "history": [
                    {"role": "user", "content": "Tell me about Python."},
                    {
                        "role": "assistant",
                        "content": "Python is a high-level programming language known for its simplicity.",
                    },
                    {"role": "user", "content": "Can you tell me more?"},
                ],
                "response": "Python is a high-level programming language known for its simplicity.",
                "human_score": 0,  # Exact repetition (bad)
            },
            # Case 3: No repetition - different topic
            {
                "history": [
                    {"role": "user", "content": "What's the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                    {"role": "user", "content": "What about Germany?"},
                ],
                "response": "The capital of Germany is Berlin.",
                "human_score": 1,  # No repetition (good)
            },
            # Case 4: Partial repetition - similar structure
            {
                "history": [
                    {"role": "user", "content": "How do I learn programming?"},
                    {"role": "assistant", "content": "Start with online tutorials and practice coding daily."},
                    {"role": "user", "content": "Any other tips?"},
                ],
                "response": "Start with online tutorials and practice coding daily. Also join coding communities.",
                "human_score": 0,  # Partial repetition (bad)
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-max-preview", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_basic_evaluation_with_runner(self, dataset, model):
        """Test the grader's basic evaluation capability"""
        # Create grader with real model
        grader = ResponseRepetitionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "response_repetition": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Check that all evaluations completed successfully
        assert len(results["response_repetition"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["response_repetition"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "response_repetition" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ResponseRepetitionGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "response_repetition_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "response_repetition_run2": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_repetition_run1"],
            another_grader_results=results["response_repetition_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        # Note: nan means perfect consistency (all scores identical)
        import math

        assert (
            math.isnan(consistency_result.consistency) or consistency_result.consistency >= 0.9
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"

    @pytest.mark.asyncio
    async def test_discriminative_power(self, dataset, model):
        """Test the grader's ability to distinguish between repetitive and non-repetitive responses"""
        # Create grader with real model
        grader = ResponseRepetitionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "response_repetition": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Calculate discriminative accuracy
        # Score >= 3 means no/minimal repetition (good), < 3 means significant repetition (bad)
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["response_repetition"]):
            # Predict: score >= 3 means good (no repetition) -> 1, score < 3 means bad (repetition) -> 0
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(
            f"\nResponseRepetition Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
        )

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
