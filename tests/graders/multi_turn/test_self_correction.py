#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for SelfCorrectionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using SelfCorrectionGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_self_correction.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_self_correction.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_self_correction.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.self_correction_grader import SelfCorrectionGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestSelfCorrectionGraderUnit:
    """Unit tests for SelfCorrectionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = SelfCorrectionGrader(model=mock_model)
        assert grader.name == "self_correction"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_good_correction(self):
        """Test evaluation when assistant corrects error properly"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant acknowledged the error and provided correct information.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = SelfCorrectionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What is the capital of Australia?"},
                {"role": "assistant", "content": "The capital of Australia is Sydney."},
                {"role": "user", "content": "That's not correct. Sydney is not the capital."},
            ]
            response = "You're right, I apologize. The capital of Australia is Canberra, not Sydney."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "self_correction"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_poor_correction(self):
        """Test evaluation when assistant fails to correct error"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Assistant insisted on the wrong answer.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = SelfCorrectionGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "What is the capital of Australia?"},
                {"role": "assistant", "content": "The capital of Australia is Sydney."},
                {"role": "user", "content": "That's not correct."},
            ]
            response = "I'm confident that Sydney is the capital of Australia."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "self_correction"
            assert result.score == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = SelfCorrectionGrader(model=mock_model)
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
class TestSelfCorrectionGraderQuality:
    """Quality tests for SelfCorrectionGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor self-correction examples"""
        return [
            # Case 1: Good correction - diabetes treatment
            {
                "history": [
                    {"role": "user", "content": "What is the most effective treatment for type 1 diabetes?"},
                    {
                        "role": "assistant",
                        "content": "Type 1 diabetes is best treated with dietary changes and exercise alone.",
                    },
                    {"role": "user", "content": "Are you sure about that? I believe that might not be accurate."},
                ],
                "response": "You are correct, and I apologize. The most effective treatment for type 1 diabetes is insulin therapy, as individuals with type 1 diabetes do not produce insulin.",
                "human_score": 1,  # Good correction
            },
            # Case 2: Good correction - pancreas function
            {
                "history": [
                    {"role": "user", "content": "What is the main function of the pancreas?"},
                    {
                        "role": "assistant",
                        "content": "The main function of the pancreas is to store and concentrate bile produced by the liver.",
                    },
                    {"role": "user", "content": "That doesn't sound right. Can you check that information again?"},
                ],
                "response": "My apologies, I misspoke. The main function of the pancreas is to produce enzymes that help digest proteins, fats, and carbohydrates. The organ that stores bile is the gallbladder.",
                "human_score": 1,  # Good correction
            },
            # Case 3: Poor correction - refuses to correct
            {
                "history": [
                    {"role": "user", "content": "What is the capital of Canada?"},
                    {"role": "assistant", "content": "The capital of Canada is Toronto."},
                    {"role": "user", "content": "That's incorrect. Toronto is not the capital."},
                ],
                "response": "Toronto is the largest and most well-known city in Canada, so many people consider it the capital.",
                "human_score": 0,  # Poor correction - didn't acknowledge error
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
        grader = SelfCorrectionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "self_correction": GraderConfig(
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
        assert len(results["self_correction"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["self_correction"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "self_correction" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = SelfCorrectionGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "self_correction_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "self_correction_run2": GraderConfig(
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
            grader_results=results["self_correction_run1"],
            another_grader_results=results["self_correction_run2"],
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
        """Test the grader's ability to distinguish between good and poor self-correction"""
        # Create grader with real model
        grader = SelfCorrectionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "self_correction": GraderConfig(
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
        # Score >= 3 means good correction, < 3 means poor correction
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["self_correction"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(f"\nSelfCorrection Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
