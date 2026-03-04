#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for TopicSwitchGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using TopicSwitchGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_topic_switch.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_topic_switch.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_topic_switch.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.topic_switch_grader import TopicSwitchGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestTopicSwitchGraderUnit:
    """Unit tests for TopicSwitchGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = TopicSwitchGrader(model=mock_model)
        assert grader.name == "topic_switch"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_good_topic_switch(self):
        """Test evaluation when topic switch is handled well"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant smoothly transitioned to the new topic.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TopicSwitchGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What's the best programming language?"},
                {"role": "assistant", "content": "Python is great for beginners."},
                {"role": "user", "content": "By the way, what's the weather like today?"},
            ]
            response = "I don't have access to real-time weather data, but you can check weather.com."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "topic_switch"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_poor_topic_switch(self):
        """Test evaluation when topic switch is handled poorly"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Assistant failed to recognize the topic change.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = TopicSwitchGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What's the best programming language?"},
                {"role": "assistant", "content": "Python is great for beginners."},
                {"role": "user", "content": "What's the weather like today?"},
            ]
            response = "Python also has great libraries for data science."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "topic_switch"
            assert result.score == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = TopicSwitchGrader(model=mock_model)
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
class TestTopicSwitchGraderQuality:
    """Quality tests for TopicSwitchGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor topic switch handling"""
        return [
            # Case 1: Good switch - from Day of the Dead to Olympics
            {
                "history": [
                    {"role": "user", "content": "What does the 'Day of the Dead' in Mexico celebrate?"},
                    {
                        "role": "assistant",
                        "content": "The 'Day of the Dead' is a Mexican holiday where families honor their deceased loved ones.",
                    },
                    {"role": "user", "content": "Which country has the most gold medals at the 2020 Olympics?"},
                ],
                "response": "At the 2020 Summer Olympics in Tokyo, the United States finished with the most gold medals, securing 39 gold medals.",
                "human_score": 1,  # Good switch
            },
            # Case 2: Good switch - from movie to tire changing
            {
                "history": [
                    {"role": "user", "content": "Can you suggest a good science fiction movie?"},
                    {
                        "role": "assistant",
                        "content": "'Dune' is a recent science fiction movie that has received critical acclaim.",
                    },
                    {"role": "user", "content": "Sure, but how do I change a flat tire?"},
                ],
                "response": "To change a flat tire: loosen the lug nuts, lift the car with a jack, remove the flat tire, place the spare, and tighten the lug nuts.",
                "human_score": 1,  # Good switch
            },
            # Case 3: Poor switch - continues old topic
            {
                "history": [
                    {"role": "user", "content": "I want to watch a film with a complex plot."},
                    {"role": "assistant", "content": "I'd recommend 'Inception' directed by Christopher Nolan."},
                    {"role": "user", "content": "What's a healthy breakfast option for someone on a keto diet?"},
                ],
                "response": "'Inception' features several layers of narrative complexity and dream manipulation.",
                "human_score": 0,  # Poor switch - ignored new topic
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
        grader = TopicSwitchGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "topic_switch": GraderConfig(
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
        assert len(results["topic_switch"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["topic_switch"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "topic_switch" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = TopicSwitchGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "topic_switch_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "topic_switch_run2": GraderConfig(
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
            grader_results=results["topic_switch_run1"],
            another_grader_results=results["topic_switch_run2"],
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
        """Test the grader's ability to distinguish between good and poor topic switching"""
        # Create grader with real model
        grader = TopicSwitchGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "topic_switch": GraderConfig(
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
        # Score >= 3 means good topic switch, < 3 means poor topic switch
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["topic_switch"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(f"\nTopicSwitch Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
