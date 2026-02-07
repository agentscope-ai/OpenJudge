# -*- coding: utf-8 -*-
"""
Complete demo test for ReflectionOutcomeUnderstandingGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ReflectionOutcomeUnderstandingGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_outcome_understanding.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_outcome_understanding.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_outcome_understanding.py -m quality
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
from openjudge.graders.agent import ReflectionOutcomeUnderstandingGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestReflectionOutcomeUnderstandingGraderUnit:
    """Unit tests for ReflectionOutcomeUnderstandingGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ReflectionOutcomeUnderstandingGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "reflection_outcome_understanding"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_good_understanding(self):
        """Test successful evaluation with good outcome understanding"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Reflection correctly understands the action outcome",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionOutcomeUnderstandingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="The drawer is now open.",
                reflection="I successfully opened the drawer.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "understand" in result.reason.lower() or "correct" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_misinterpretation(self):
        """Test evaluation detecting outcome misinterpretation"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Reflection misinterprets the action outcome - claims success when action failed",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionOutcomeUnderstandingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="The drawer is still closed. Action failed.",
                reflection="I successfully opened the drawer.",
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "misinterpret" in result.reason.lower() or "failed" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.85,  # Will be normalized to 1.0
            "reason": "Reflection correctly understands outcome given the context",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionOutcomeUnderstandingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "Drawer is locked", "reflection": "Need to find key"},
            ]

            result = await grader.aevaluate(
                observation="The drawer is still locked. Cannot open it.",
                reflection="The drawer remains locked because I don't have the key yet.",
                task_context="Task: Open the drawer",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "understand" in result.reason.lower() or "correct" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ReflectionOutcomeUnderstandingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Action completed.",
                reflection="Task done.",
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


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestReflectionOutcomeUnderstandingGraderQuality:
    """Quality tests for ReflectionOutcomeUnderstandingGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real reflection outcome misinterpretation cases"""
        return [
            # Case 1: Good understanding - correctly interprets success
            {
                "observation": "The drawer is now open.",
                "reflection": "I successfully opened the drawer.",
                "task_context": "Task: Open the drawer",
                "history_steps": [],
                "human_score": 1,  # Good understanding
            },
            # Case 2: Misinterpretation - claims success when failed
            {
                "observation": "The drawer is still closed. Action failed.",
                "reflection": "I successfully opened the drawer.",
                "task_context": "Task: Open the drawer",
                "history_steps": [],
                "human_score": 0,  # Poor understanding
            },
            # Case 3: Good understanding - recognizes failure
            {
                "observation": "The door is still locked. Cannot open it.",
                "reflection": "My attempt to open the door failed because it's locked.",
                "task_context": "Task: Open the door",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 4: Misinterpretation - claims door is open when locked
            {
                "observation": "The door is still locked.",
                "reflection": "The door is now open and I can proceed.",
                "task_context": "Task: Open the door",
                "history_steps": [],
                "human_score": 0,
            },
            # Case 5: Good understanding - recognizes no change
            {
                "observation": "Nothing happened. The cabinet remains closed.",
                "reflection": "The action had no effect. The cabinet is still closed.",
                "task_context": "Task: Open cabinet",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 6: Misinterpretation - overestimates outcome
            {
                "observation": "Nothing happened. The cabinet remains closed.",
                "reflection": "Great! I opened the cabinet and can see inside now.",
                "task_context": "Task: Open cabinet",
                "history_steps": [],
                "human_score": 0,
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
        """Test the grader's ability to distinguish between good and poor outcome understanding"""
        # Create grader with real model
        grader = ReflectionOutcomeUnderstandingGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "reflection_outcome_understanding": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["reflection_outcome_understanding"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.7, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ReflectionOutcomeUnderstandingGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "reflection_outcome_understanding_run1": GraderConfig(
                grader=grader,
            ),
            "reflection_outcome_understanding_run2": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["reflection_outcome_understanding_run1"],
            another_grader_results=results["reflection_outcome_understanding_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.7
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestReflectionOutcomeUnderstandingGraderAdversarial:
    """Adversarial tests for ReflectionOutcomeUnderstandingGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with good and poor understanding pairs"""
        return [
            {
                "observation": "The drawer is still closed. Action failed.",
                "good_reflection": "My attempt to open the drawer failed. It's still closed.",
                "poor_reflection": "I successfully opened the drawer.",
                "task_context": "Task: Open the drawer",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "The door is still locked.",
                "good_reflection": "The door remains locked. I cannot open it yet.",
                "poor_reflection": "The door is now open and I can proceed.",
                "task_context": "Task: Open the door",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Nothing happened. The cabinet remains closed.",
                "good_reflection": "The action had no effect. The cabinet is still closed.",
                "poor_reflection": "Great! I opened the cabinet and can see inside now.",
                "task_context": "Task: Open cabinet",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "You tried to pick up the key, but it's too far to reach.",
                "good_reflection": "I couldn't reach the key. Need to find another approach.",
                "poor_reflection": "I successfully picked up the key and added it to inventory.",
                "task_context": "Task: Get the key",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "The box is empty. Nothing inside.",
                "good_reflection": "I checked the box but found nothing inside.",
                "poor_reflection": "Found valuable items in the box!",
                "task_context": "Task: Search for items",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
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
    async def test_adversarial_reflection_outcome_understanding_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ReflectionOutcomeUnderstandingGrader(model=model)

        # Configure GraderConfig to evaluate both good and poor reflections
        grader_configs = {
            "reflection_outcome_understanding_good": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "reflection": "good_reflection",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "reflection_outcome_understanding_poor": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "reflection": "poor_reflection",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for poor reflections
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["reflection_outcome_understanding_poor"],
            label_path="poor_label",
        )

        # Use FalseNegativeAnalyzer for good reflections
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["reflection_outcome_understanding_good"],
            label_path="good_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
