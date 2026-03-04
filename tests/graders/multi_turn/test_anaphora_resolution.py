#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for AnaphoraResolutionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using AnaphoraResolutionGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_anaphora_resolution.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_anaphora_resolution.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_anaphora_resolution.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.anaphora_resolution_grader import (
    AnaphoraResolutionGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestAnaphoraResolutionGraderUnit:
    """Unit tests for AnaphoraResolutionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = AnaphoraResolutionGrader(model=mock_model)
        assert grader.name == "anaphora_resolution"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_correct_resolution(self):
        """Test evaluation when pronouns are correctly resolved"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant correctly identified that 'it' refers to the laptop.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = AnaphoraResolutionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I bought a new laptop yesterday."},
                {"role": "assistant", "content": "That's great! What brand is it?"},
                {"role": "user", "content": "It's a MacBook Pro. Is it good for programming?"},
            ]
            response = "Yes, the MacBook Pro is excellent for programming."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "anaphora_resolution"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_incorrect_resolution(self):
        """Test evaluation when pronouns are incorrectly resolved"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Assistant misunderstood the reference.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = AnaphoraResolutionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "I bought a new laptop yesterday."},
                {"role": "assistant", "content": "That's great!"},
                {"role": "user", "content": "Is it good for programming?"},
            ]
            response = "Programming is a great skill to learn!"

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "anaphora_resolution"
            assert result.score == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = AnaphoraResolutionGrader(model=mock_model)
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
class TestAnaphoraResolutionGraderQuality:
    """Quality tests for AnaphoraResolutionGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor anaphora resolution examples"""
        return [
            # Case 1: Good resolution - correctly understands "it" refers to Turing Test
            {
                "history": [
                    {"role": "user", "content": "What are the implications of the Turing Test for AI?"},
                    {
                        "role": "assistant",
                        "content": "The Turing Test is significant for AI as it provides a benchmark for human-like intelligence.",
                    },
                    {"role": "user", "content": "And what does it say about the machine if it fails the test?"},
                ],
                "response": "If the machine fails the Turing Test, it indicates that its conversational capabilities are not yet sophisticated enough to fool a human.",
                "human_score": 1,  # Good resolution
            },
            # Case 2: Good resolution - correctly understands "they" refers to parameters
            {
                "history": [
                    {"role": "user", "content": "Can you explain what machine learning is?"},
                    {
                        "role": "assistant",
                        "content": "Machine learning enables computers to learn from data by adjusting parameters within an algorithm.",
                    },
                    {"role": "user", "content": "And these parameters, are they set by the programmers initially?"},
                ],
                "response": "Yes, programmers set initial values for the parameters, which are then adjusted during the training process.",
                "human_score": 1,  # Good resolution
            },
            # Case 3: Poor resolution - misunderstands the reference
            {
                "history": [
                    {"role": "user", "content": "I'm reading about quantum computing."},
                    {"role": "assistant", "content": "Quantum computing uses qubits instead of classical bits."},
                    {"role": "user", "content": "How fast is it compared to classical computers?"},
                ],
                "response": "Classical computers have been around for decades and continue to improve in speed.",
                "human_score": 0,  # Poor resolution - didn't address quantum computing
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
        grader = AnaphoraResolutionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "anaphora_resolution": GraderConfig(
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
        assert len(results["anaphora_resolution"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["anaphora_resolution"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "anaphora_resolution" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = AnaphoraResolutionGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "anaphora_resolution_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "anaphora_resolution_run2": GraderConfig(
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
            grader_results=results["anaphora_resolution_run1"],
            another_grader_results=results["anaphora_resolution_run2"],
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
        """Test the grader's ability to distinguish between good and poor anaphora resolution"""
        # Create grader with real model
        grader = AnaphoraResolutionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "anaphora_resolution": GraderConfig(
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
        # Score >= 3 means good resolution, < 3 means poor resolution
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["anaphora_resolution"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(
            f"\nAnaphoraResolution Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
        )

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
