#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ReflectionProgressAwarenessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ReflectionProgressAwarenessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_progress_awareness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_progress_awareness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/reflection/test_reflection_progress_awareness.py -m quality
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
from openjudge.graders.agent import ReflectionProgressAwarenessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestReflectionProgressAwarenessGraderUnit:
    """Unit tests for ReflectionProgressAwarenessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ReflectionProgressAwarenessGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "reflection_progress_awareness"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_good_awareness(self):
        """Test successful evaluation with good progress awareness"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Reflection accurately assesses progress toward the goal",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionProgressAwarenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Found 3 apples in cabinet 2.",
                reflection="Good progress! Found apples as required.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "progress" in result.reason.lower() or "accurate" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_overestimation(self):
        """Test evaluation detecting progress overestimation"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Reflection overestimates progress - claims excellent progress despite finding nothing",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionProgressAwarenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Cabinet 1 is still empty. No items found.",
                reflection="Excellent progress! I'm making great headway toward finding the apples!",
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "overestimat" in result.reason.lower() or "excellent" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.85,  # Will be normalized to 1.0
            "reason": "Reflection accurately assesses lack of progress given repeated failures",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReflectionProgressAwarenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "Cabinet 1 empty", "reflection": "No items"},
                {"step": 2, "observation": "Cabinet 2 empty", "reflection": "Still nothing"},
            ]

            result = await grader.aevaluate(
                observation="Cabinet 3 is also empty.",
                reflection="No progress yet. Need to keep searching.",
                task_context="Task: Find items",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "progress" in result.reason.lower() or "accurate" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ReflectionProgressAwarenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Action completed.",
                reflection="Making progress.",
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
class TestReflectionProgressAwarenessGraderQuality:
    """Quality tests for ReflectionProgressAwarenessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real reflection progress misjudge cases"""
        return [
            # Case 1: Good awareness - recognizes positive progress
            {
                "observation": "Found 3 apples in cabinet 2.",
                "reflection": "Good progress! Found apples as required.",
                "task_context": "Task: Find apples",
                "history_steps": [],
                "human_score": 1,  # Good awareness
            },
            # Case 2: Overestimation - claims excellent progress despite no findings
            {
                "observation": "Cabinet 1 is still empty. No items found.",
                "reflection": "Excellent progress! I'm making great headway toward finding the apples!",
                "task_context": "Task: Find apples in cabinets",
                "history_steps": [],
                "human_score": 0,  # Poor awareness (overestimation)
            },
            # Case 3: Good awareness - recognizes lack of progress
            {
                "observation": "The door is still locked. Cannot proceed.",
                "reflection": "No progress. Still stuck at the locked door.",
                "task_context": "Task: Open the door and proceed",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 4: Overestimation - claims progress when stuck
            {
                "observation": "The door is still locked. No key found.",
                "reflection": "Making great progress toward completing the task!",
                "task_context": "Task: Unlock and open the door",
                "history_steps": [],
                "human_score": 0,
            },
            # Case 5: Good awareness - realistic assessment of partial progress
            {
                "observation": "Found 2 items. Still need 3 more to complete the task.",
                "reflection": "Made some progress but still have more work to do.",
                "task_context": "Task: Collect 5 items",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 6: Overestimation - exaggerates partial progress
            {
                "observation": "Found 1 item. Still need 9 more.",
                "reflection": "Almost done! Task nearly complete!",
                "task_context": "Task: Collect 10 items",
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
        """Test the grader's ability to distinguish between good and poor progress awareness"""
        # Create grader with real model
        grader = ReflectionProgressAwarenessGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "reflection_progress_awareness": GraderConfig(
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
            grader_results=results["reflection_progress_awareness"],
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
        grader = ReflectionProgressAwarenessGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "reflection_progress_awareness_run1": GraderConfig(
                grader=grader,
            ),
            "reflection_progress_awareness_run2": GraderConfig(
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
            grader_results=results["reflection_progress_awareness_run1"],
            another_grader_results=results["reflection_progress_awareness_run2"],
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
class TestReflectionProgressAwarenessGraderAdversarial:
    """Adversarial tests for ReflectionProgressAwarenessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with good and poor awareness pairs"""
        return [
            {
                "observation": "Cabinet 1 is still empty. No items found.",
                "good_reflection": "No progress yet. This cabinet is empty.",
                "poor_reflection": "Excellent progress! I'm making great headway toward finding the apples!",
                "task_context": "Task: Find apples in cabinets",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "The door is still locked. No key found.",
                "good_reflection": "No progress. Still stuck at the locked door without a key.",
                "poor_reflection": "Making great progress toward completing the task!",
                "task_context": "Task: Unlock and open the door",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Found 1 item. Still need 9 more.",
                "good_reflection": "Found one item, but still have a long way to go.",
                "poor_reflection": "Almost done! Task nearly complete!",
                "task_context": "Task: Collect 10 items",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Checked 3 rooms. All empty. No target found.",
                "good_reflection": "No progress finding the target. Need to search more locations.",
                "poor_reflection": "Excellent! Getting very close to finding the target!",
                "task_context": "Task: Find the target object",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Action failed. Nothing changed.",
                "good_reflection": "My action failed and made no progress.",
                "poor_reflection": "Great success! Moving forward nicely!",
                "task_context": "Task: Complete the objective",
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
    async def test_adversarial_reflection_progress_awareness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ReflectionProgressAwarenessGrader(model=model)

        # Configure GraderConfig to evaluate both good and poor reflections
        grader_configs = {
            "reflection_progress_awareness_good": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "reflection": "good_reflection",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "reflection_progress_awareness_poor": GraderConfig(
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
            grader_results=results["reflection_progress_awareness_poor"],
            label_path="poor_label",
        )

        # Use FalseNegativeAnalyzer for good reflections
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["reflection_progress_awareness_good"],
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
