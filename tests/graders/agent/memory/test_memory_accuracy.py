#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for MemoryAccuracyGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using MemoryAccuracyGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_accuracy.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_accuracy.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/memory/test_memory_accuracy.py -m quality
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
from openjudge.graders.agent import MemoryAccuracyGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestMemoryAccuracyGraderUnit:
    """Unit tests for MemoryAccuracyGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = MemoryAccuracyGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "memory_accuracy"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_accurate(self):
        """Test successful evaluation with accurate memory"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Memory accurately reflects the observation",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Cabinet 1 contains 3 red apples.",
                memory="Cabinet 1 has 3 red apples.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "accurate" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_hallucination(self):
        """Test evaluation detecting memory hallucination"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Memory contains hallucinated information not present in observation",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="You see a closed cabinet.",
                memory="There is a red vase inside the cabinet with gold trim.",
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "hallucinated" in result.reason.lower() or "not present" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.8,  # Will be normalized to 1.0
            "reason": "Memory is accurate given the observation and history context",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "You enter a room", "memory": "I am in a room"},
            ]

            result = await grader.aevaluate(
                observation="You see a table with a key on it.",
                memory="There is a key on the table in the room.",
                task_context="Task: Find the key to unlock the door",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "accurate" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = MemoryAccuracyGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="You see a closed cabinet.",
                memory="Cabinet is closed.",
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
class TestMemoryAccuracyGraderQuality:
    """Quality tests for MemoryAccuracyGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real memory hallucination cases"""
        return [
            # Case 1: Accurate memory - correctly reflects observation
            {
                "observation": "Cabinet 1 contains 3 red apples.",
                "memory": "Cabinet 1 has 3 red apples.",
                "task_context": "Task: Inventory items in the room",
                "history_steps": [],
                "human_score": 1,  # Accurate memory
            },
            # Case 2: Hallucination - claims to see inside closed cabinet
            {
                "observation": "You see a closed cabinet.",
                "memory": "There is a red vase inside the cabinet with gold trim.",
                "task_context": "Task: Inventory room objects",
                "history_steps": [],
                "human_score": 0,  # Hallucinated information
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
        """Test the grader's ability to distinguish between accurate and hallucinated memory"""
        # Create grader with real model
        grader = MemoryAccuracyGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "memory_accuracy": GraderConfig(
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
            grader_results=results["memory_accuracy"],
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
        grader = MemoryAccuracyGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "memory_accuracy_run1": GraderConfig(
                grader=grader,
            ),
            "memory_accuracy_run2": GraderConfig(
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
            grader_results=results["memory_accuracy_run1"],
            another_grader_results=results["memory_accuracy_run2"],
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
class TestMemoryAccuracyGraderAdversarial:
    """Adversarial tests for MemoryAccuracyGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with accurate and hallucinated pairs"""
        return [
            {
                "observation": "You see a closed cabinet.",
                "accurate_memory": "The cabinet is closed.",
                "hallucinated_memory": "There is a red vase inside the cabinet with gold trim.",
                "task_context": "Task: Inventory room objects",
                "history_steps": [],
                "accurate_label": 1,
                "hallucinated_label": 0,
            },
            {
                "observation": "Cabinet is still locked. Cannot see inside.",
                "accurate_memory": "Cabinet 1 is locked and I cannot see inside.",
                "hallucinated_memory": "Cabinet contains 5 golden coins.",
                "task_context": "Task: Find items in the room",
                "history_steps": [
                    {"step": 1, "observation": "Cabinet is locked", "memory": "Cabinet 1 is locked"},
                ],
                "accurate_label": 1,
                "hallucinated_label": 0,
            },
            {
                "observation": "You open drawer 1. Inside you see a key.",
                "accurate_memory": "Drawer 1 contains a key.",
                "hallucinated_memory": "Drawer 1 contains a key, a pen, and a notebook.",
                "task_context": "Task: Search for the key",
                "history_steps": [],
                "accurate_label": 1,
                "hallucinated_label": 0,
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
    async def test_adversarial_memory_accuracy_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = MemoryAccuracyGrader(model=model)

        # Configure GraderConfig to evaluate both accurate and hallucinated memories
        grader_configs = {
            "memory_accuracy_accurate": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "memory": "accurate_memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "memory_accuracy_hallucinated": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "memory": "hallucinated_memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for hallucinated memories
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_accuracy_hallucinated"],
            label_path="hallucinated_label",
        )

        # Use FalseNegativeAnalyzer for accurate memories
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_accuracy_accurate"],
            label_path="accurate_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.3, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
