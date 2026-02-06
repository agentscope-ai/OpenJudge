#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ProactiveInteractionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ProactiveInteractionGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_proactive_interaction.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_proactive_interaction.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_proactive_interaction.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.proactive_interaction_grader import (
    ProactiveInteractionGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestProactiveInteractionGraderUnit:
    """Unit tests for ProactiveInteractionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ProactiveInteractionGrader(model=mock_model)
        assert grader.name == "proactive_interaction"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_good_proactive_interaction(self):
        """Test evaluation when assistant proactively engages"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant proactively asked follow-up questions.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ProactiveInteractionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I just watched a great movie last night."},
            ]
            response = "That sounds exciting! What movie did you watch? I'd love to hear what you thought about it."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "proactive_interaction"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_poor_proactive_interaction(self):
        """Test evaluation when assistant is passive"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Assistant gave a minimal response without engaging.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ProactiveInteractionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I just watched a great movie last night."},
            ]
            response = "That's nice."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "proactive_interaction"
            assert result.score == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ProactiveInteractionGrader(model=mock_model)
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
class TestProactiveInteractionGraderQuality:
    """Quality tests for ProactiveInteractionGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor proactive interaction examples"""
        return [
            # Case 1: Good proactive - asks about stress management
            {
                "history": [
                    {"role": "user", "content": "Lately, I've been feeling overwhelmed with stress."},
                    {
                        "role": "assistant",
                        "content": "Feeling overwhelmed can be tough. What do you think has been the primary cause of your stress?",
                    },
                    {"role": "user", "content": "Work has been extremely busy."},
                    {
                        "role": "assistant",
                        "content": "That sounds like a lot. Have you tried any strategies to prioritize your tasks?",
                    },
                    {"role": "user", "content": "I've made some to-do lists, but they just keep getting longer."},
                ],
                "response": "I understand how that can be discouraging. Have you set aside any time for self-care to help alleviate the stress? Shall we explore some self-care techniques that might work for your situation?",
                "human_score": 1,  # Good proactive
            },
            # Case 2: Good proactive - movie discussion
            {
                "history": [
                    {"role": "user", "content": "I watched a fantastic sci-fi movie last night."},
                ],
                "response": "That's intriguing! What's the name of the movie? I'd love to hear more about what made it fantastic for you.",
                "human_score": 1,  # Good proactive
            },
            # Case 3: Poor proactive - minimal engagement
            {
                "history": [
                    {"role": "user", "content": "I'm thinking about learning a new language."},
                ],
                "response": "That's a good idea.",
                "human_score": 0,  # Poor - no follow-up questions
            },
            # Case 4: Good proactive - restaurant recommendation
            {
                "history": [
                    {"role": "user", "content": "Last night, I had the best sushi of my life at a new place."},
                ],
                "response": "Sushi with a view sounds amazing! What made the sushi the best you've ever had? Did the restaurant have any specialty rolls?",
                "human_score": 1,  # Good proactive
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
        grader = ProactiveInteractionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "proactive_interaction": GraderConfig(
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
        assert len(results["proactive_interaction"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["proactive_interaction"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "proactive_interaction" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ProactiveInteractionGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "proactive_interaction_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "proactive_interaction_run2": GraderConfig(
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
            grader_results=results["proactive_interaction_run1"],
            another_grader_results=results["proactive_interaction_run2"],
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
        """Test the grader's ability to distinguish between good and poor proactive interaction"""
        # Create grader with real model
        grader = ProactiveInteractionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "proactive_interaction": GraderConfig(
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
        # Score >= 3 means good proactive interaction, < 3 means poor
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["proactive_interaction"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(
            f"\nProactiveInteraction Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
        )

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
