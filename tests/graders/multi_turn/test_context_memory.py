#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ContextMemoryGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ContextMemoryGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_context_memory.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_context_memory.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_context_memory.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.context_memory_grader import ContextMemoryGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestContextMemoryGraderUnit:
    """Unit tests for ContextMemoryGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ContextMemoryGrader(model=mock_model)
        assert grader.name == "context_memory"
        assert grader.model == mock_model

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()

        grader_zh = ContextMemoryGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

        grader_en = ContextMemoryGrader(model=mock_model, language=LanguageEnum.EN)
        assert grader_en.language == LanguageEnum.EN

    @pytest.mark.asyncio
    async def test_evaluate_with_good_memory(self):
        """Test evaluation when assistant remembers context well"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant accurately remembered all key details from earlier conversation.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ContextMemoryGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I'm a vegetarian looking for recipes."},
                {"role": "assistant", "content": "I'd recommend vegetable lasagna!"},
                {"role": "user", "content": "Can I use zucchini?"},
            ]
            response = "Yes, zucchini works great in vegetarian lasagna!"

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "context_memory"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_with_poor_memory(self):
        """Test evaluation when assistant forgets context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Assistant forgot the vegetarian constraint and suggested beef.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ContextMemoryGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I'm a vegetarian looking for recipes."},
                {"role": "assistant", "content": "I'd recommend vegetable lasagna!"},
                {"role": "user", "content": "What protein should I add?"},
            ]
            response = "You could add ground beef for extra protein!"

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "context_memory"
            assert result.score == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ContextMemoryGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [{"role": "user", "content": "Hello"}]
            result = await grader.aevaluate(response="Hi there!", history=history)

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
class TestContextMemoryGraderQuality:
    """Quality tests for ContextMemoryGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor context memory examples"""
        return [
            # Case 1: Good memory - remembers user preference
            {
                "history": [
                    {"role": "user", "content": "I'm trying to pick a movie to watch tonight. Can you help?"},
                    {
                        "role": "assistant",
                        "content": "Of course, I'd be happy to help. What type of movie are you in the mood for?",
                    },
                    {"role": "user", "content": "I love a good mystery."},
                    {
                        "role": "assistant",
                        "content": "Great! Some popular mystery films include 'Se7en', 'The Girl with the Dragon Tattoo', and 'Shutter Island'.",
                    },
                    {
                        "role": "user",
                        "content": "I've watched 'Se7en' and 'Shutter Island'. Who directed 'The Girl with the Dragon Tattoo'?",
                    },
                ],
                "response": "'The Girl with the Dragon Tattoo' was directed by David Fincher, who is known for his mastery in creating suspenseful movies.",
                "human_score": 1,  # Good memory
            },
            # Case 2: Good memory - remembers budget constraint
            {
                "history": [
                    {"role": "user", "content": "I want to buy a new laptop for my graphic design work."},
                    {
                        "role": "assistant",
                        "content": "For graphic design, you might want to consider high-resolution laptops. Some options include MacBook Pro, Dell XPS 15.",
                    },
                    {"role": "user", "content": "My budget is under $1500."},
                    {"role": "assistant", "content": "The Dell XPS 15 could be a good fit for your budget."},
                    {
                        "role": "user",
                        "content": "Which one would be the best option considering my budget and requirements?",
                    },
                ],
                "response": "Taking into account your budget of under $1500 and the necessity of a good display for graphic design, I would recommend the Dell XPS 15.",
                "human_score": 1,  # Good memory - remembered budget
            },
            # Case 3: Poor memory - forgets vegetarian constraint
            {
                "history": [
                    {"role": "user", "content": "I'm a vegetarian and I'm looking for some new recipes."},
                    {"role": "assistant", "content": "How about trying a vegetable lasagna?"},
                    {"role": "user", "content": "Sounds good. Can I use zucchini and bell peppers?"},
                    {"role": "assistant", "content": "Absolutely, both add wonderful flavor to the lasagna."},
                    {"role": "user", "content": "What protein should I add to make it more filling?"},
                ],
                "response": "You could add ground beef or chicken for extra protein!",
                "human_score": 0,  # Poor memory - forgot vegetarian
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
        grader = ContextMemoryGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "context_memory": GraderConfig(
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
        assert len(results["context_memory"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["context_memory"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "context_memory" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ContextMemoryGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "context_memory_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "context_memory_run2": GraderConfig(
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
            grader_results=results["context_memory_run1"],
            another_grader_results=results["context_memory_run2"],
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
        """Test the grader's ability to distinguish between good and poor context memory"""
        # Create grader with real model
        grader = ContextMemoryGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "context_memory": GraderConfig(
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
        # Score >= 3 means good memory, < 3 means poor memory
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["context_memory"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(f"\nContextMemory Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
