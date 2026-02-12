#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for InstructionClarificationGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using InstructionClarificationGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multi_turn/test_instruction_clarification.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multi_turn/test_instruction_clarification.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multi_turn/test_instruction_clarification.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.graders.multi_turn.instruction_clarification_grader import (
    InstructionClarificationGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestInstructionClarificationGraderUnit:
    """Unit tests for InstructionClarificationGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = InstructionClarificationGrader(model=mock_model)
        assert grader.name == "instruction_clarification"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_good_clarification(self):
        """Test evaluation when assistant asks for clarification appropriately"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Assistant appropriately asked for clarification.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = InstructionClarificationGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "How long does it take for medicine to work?"},
            ]
            response = "The onset of action can vary greatly depending on the type of medicine. Could you specify which medicine you are referring to?"

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "instruction_clarification"
            assert result.score == 5

    @pytest.mark.asyncio
    async def test_evaluate_poor_clarification(self):
        """Test evaluation when assistant fails to ask for needed clarification"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Assistant made assumptions instead of asking for clarification.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = InstructionClarificationGrader(model=mock_model, language=LanguageEnum.EN)
            grader.model.achat = mock_achat

            history = [
                {"role": "user", "content": "How long does it take for medicine to work?"},
            ]
            response = "Medicine usually takes about 30 minutes to work."

            result = await grader.aevaluate(response=response, history=history)

            assert result.name == "instruction_clarification"
            assert result.score == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = InstructionClarificationGrader(model=mock_model)
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
class TestInstructionClarificationGraderQuality:
    """Quality tests for InstructionClarificationGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with good and poor instruction clarification examples"""
        return [
            # Case 1: Good clarification - asks about medicine type
            {
                "history": [
                    {"role": "user", "content": "How long does it take for medicine to work?"},
                ],
                "response": "The onset of action for medication can vary greatly depending on the type of medicine. Could you specify which medicine you are referring to?",
                "human_score": 1,  # Good clarification
            },
            # Case 2: Good clarification - asks about specific drug
            {
                "history": [
                    {"role": "user", "content": "What side effects are associated with this drug?"},
                ],
                "response": "To provide accurate information, could you please tell me the name of the drug you are inquiring about?",
                "human_score": 1,  # Good clarification
            },
            # Case 3: Poor clarification - makes assumptions
            {
                "history": [
                    {"role": "user", "content": "Is it safe to take medicine with alcohol?"},
                ],
                "response": "No, you should never mix medicine with alcohol as it can be dangerous.",
                "human_score": 0,  # Poor - should ask which medicine
            },
            # Case 4: Good clarification - asks about war
            {
                "history": [
                    {"role": "user", "content": "When did the war end?"},
                ],
                "response": "To provide an accurate answer, I need to know which war you are referring to. Could you please specify the conflict?",
                "human_score": 1,  # Good clarification
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
        grader = InstructionClarificationGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "instruction_clarification": GraderConfig(
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
        assert len(results["instruction_clarification"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["instruction_clarification"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "instruction_clarification" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = InstructionClarificationGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "instruction_clarification_run1": GraderConfig(
                grader=grader,
                mapper={
                    "response": "response",
                    "history": "history",
                },
            ),
            "instruction_clarification_run2": GraderConfig(
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
            grader_results=results["instruction_clarification_run1"],
            another_grader_results=results["instruction_clarification_run2"],
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
        """Test the grader's ability to distinguish between good and poor clarification"""
        # Create grader with real model
        grader = InstructionClarificationGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "instruction_clarification": GraderConfig(
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
        # Score >= 3 means good clarification, < 3 means poor clarification
        correct_predictions = 0
        total_predictions = len(dataset)

        for sample, result in zip(dataset, results["instruction_clarification"]):
            predicted = 1 if result.score >= 3 else 0
            if predicted == sample["human_score"]:
                correct_predictions += 1

            print(f"Score: {result.score}, Expected: {sample['human_score']}, Predicted: {predicted}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(
            f"\nInstructionClarification Discriminative Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
        )

        # Assert minimum accuracy threshold
        assert accuracy >= 0.5, f"Discriminative accuracy too low: {accuracy}"
