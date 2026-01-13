#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ActionAlignmentGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ActionAlignmentGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/action/test_action_alignment.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/action/test_action_alignment.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/action/test_action_alignment.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import (
    AccuracyAnalyzer,
    FalseNegativeAnalyzer,
    FalsePositiveAnalyzer,
)
from openjudge.graders.agent import ActionAlignmentGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestActionAlignmentGraderUnit:
    """Unit tests for ActionAlignmentGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ActionAlignmentGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "action_alignment"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_aligned(self):
        """Test successful evaluation with good alignment"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.8,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Action aligns well with the plan",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ActionAlignmentGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will open drawer 1 to find the key",
                action="open drawer 1",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.8
            assert "align" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_context(self):
        """Test evaluation with task context"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0
            "reason": "Action perfectly aligns with plan given the context",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ActionAlignmentGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will search for the key in drawer 1",
                action="open drawer 1",
                context="Task: Find the key to unlock the door",
            )

            # Assertions
            assert result.score == 1.0
            assert "align" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ActionAlignmentGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will open drawer 1",
                action="open drawer 1",
            )

            # Assertions
            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason


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
class TestActionAlignmentGraderQuality:
    """Quality tests for ActionAlignmentGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real GAIA error cases"""
        return [
            # Case 1: Good alignment - Opening drawer as planned
            {
                "plan": "I need to search for the key. I'll start by opening drawer 1 to look inside.",
                "action": "open drawer 1",
                "context": "Task: Find the key hidden in one of the drawers in the room",
                "human_score": 1,  # Good alignment
            },
            # Case 2: Misalignment - Closing instead of opening
            {
                "plan": "I will open drawer 1 to find the key",
                "action": "close drawer 1",
                "context": "Task: Find the key to unlock the door",
                "human_score": 0,  # Poor alignment
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
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between aligned and misaligned actions"""
        # Create grader with real model
        grader = ActionAlignmentGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "action_alignment": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "plan",
                    "action": "action",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["action_alignment"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.5, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

        # Print accuracy for debugging
        print(f"Accuracy: {accuracy_result.accuracy}")

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ActionAlignmentGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "action_alignment_run1": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "plan",
                    "action": "action",
                    "context": "context",
                },
            ),
            "action_alignment_run2": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "plan",
                    "action": "action",
                    "context": "context",
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
            grader_results=results["action_alignment_run1"],
            another_grader_results=results["action_alignment_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.9
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestActionAlignmentGraderAdversarial:
    """Adversarial tests for ActionAlignmentGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with aligned and misaligned pairs"""
        return [
            {
                "plan": "I will search for the key in drawer 1",
                "aligned_action": "open drawer 1",
                "misaligned_action": "close drawer 1",
                "context": "Task: Find the key to unlock the door",
                "aligned_label": 1,
                "misaligned_label": 0,
            },
            {
                "plan": "First I need to examine the desk to see what's available",
                "aligned_action": "examine desk",
                "misaligned_action": "take lamp",
                "context": "Task: Organize the desk",
                "aligned_label": 1,
                "misaligned_label": 0,
            },
            {
                "plan": "I should pick up the red ball from the floor",
                "aligned_action": "take red ball",
                "misaligned_action": "take blue cube",
                "context": "Task: Collect red objects",
                "aligned_label": 1,
                "misaligned_label": 0,
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
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_action_alignment_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ActionAlignmentGrader(model=model)

        # Configure GraderConfig to evaluate both aligned and misaligned actions
        grader_configs = {
            "action_alignment_aligned": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "plan",
                    "action": "aligned_action",
                    "context": "context",
                },
            ),
            "action_alignment_misaligned": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "plan",
                    "action": "misaligned_action",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for misaligned actions
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["action_alignment_misaligned"],
            label_path="misaligned_label",
        )

        # Use FalseNegativeAnalyzer for aligned actions
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["action_alignment_aligned"],
            label_path="aligned_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.3, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
