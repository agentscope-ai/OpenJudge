#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for MemoryDetailPreservationGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using MemoryDetailPreservationGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_detail_preservation.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_detail_preservation.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/memory/test_memory_detail_preservation.py -m quality
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
from openjudge.graders.agent import MemoryDetailPreservationGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestMemoryDetailPreservationGraderUnit:
    """Unit tests for MemoryDetailPreservationGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = MemoryDetailPreservationGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "memory_detail_preservation"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_good_preservation(self):
        """Test successful evaluation with good detail preservation"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Memory preserves all important details from the observation",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryDetailPreservationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
                memory="Cabinet 1 at coordinates (3.5, 2.1): 5 red apples.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "detail" in result.reason.lower() or "preserve" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_oversimplification(self):
        """Test evaluation detecting memory over-simplification"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Memory over-simplifies the observation, losing critical details like coordinates and quantities",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryDetailPreservationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
                memory="Found some apples in a cabinet.",  # Too vague
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "simplif" in result.reason.lower() or "detail" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.8,  # Will be normalized to 1.0
            "reason": "Memory preserves key details given the task context",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryDetailPreservationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "You enter a room", "memory": "I am in a room"},
            ]

            result = await grader.aevaluate(
                observation="Drawer 5 in Room B at position (10.2, 5.8) contains key-A123.",
                memory="Drawer 5 in Room B at (10.2, 5.8): key-A123.",
                task_context="Task: Locate items with precise positions",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "detail" in result.reason.lower() or "preserve" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = MemoryDetailPreservationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                observation="Cabinet contains items.",
                memory="Items in cabinet.",
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
class TestMemoryDetailPreservationGraderQuality:
    """Quality tests for MemoryDetailPreservationGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real memory over-simplification cases"""
        return [
            # Case 1: Good detail preservation - includes all key details
            {
                "observation": "Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples and 3 green apples.",
                "memory": "Cabinet 1 at (3.5, 2.1): 5 red apples, 3 green apples.",
                "task_context": "Task: Inventory items with precise locations and quantities",
                "history_steps": [],
                "human_score": 1,  # Good detail preservation
            },
            # Case 2: Over-simplification - lost critical details
            {
                "observation": "Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples and 3 green apples.",
                "memory": "Found some apples in a cabinet.",  # Too vague
                "task_context": "Task: Inventory items with precise locations and quantities",
                "history_steps": [],
                "human_score": 0,  # Poor detail preservation (over-simplified)
            },
            # Case 3: Good preservation with coordinates
            {
                "observation": "Drawer 5 in Room B at position (10.2, 5.8) contains key-A123 with gold trim.",
                "memory": "Drawer 5, Room B, position (10.2, 5.8): key-A123, gold trim.",
                "task_context": "Task: Locate specific items with exact positions",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 4: Over-simplification losing critical details
            {
                "observation": "Drawer 5 in Room B at position (10.2, 5.8) contains key-A123 with gold trim.",
                "memory": "Found a key somewhere.",  # Lost all critical details
                "task_context": "Task: Locate specific items with exact positions",
                "history_steps": [],
                "human_score": 0,
            },
            # Case 5: Good preservation with quantities
            {
                "observation": "On the shelf at height 1.5m, there are 12 red books, 8 blue notebooks, and 3 pencils.",
                "memory": "Shelf at 1.5m height: 12 red books, 8 blue notebooks, 3 pencils.",
                "task_context": "Task: Count and record items",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 6: Lost quantity details
            {
                "observation": "On the shelf at height 1.5m, there are 12 red books, 8 blue notebooks, and 3 pencils.",
                "memory": "Found books and notebooks on a shelf.",  # Lost quantities and specifics
                "task_context": "Task: Count and record items",
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
        """Test the grader's ability to distinguish between good and poor detail preservation"""
        # Create grader with real model
        grader = MemoryDetailPreservationGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "memory_detail_preservation": GraderConfig(
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
            grader_results=results["memory_detail_preservation"],
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
        grader = MemoryDetailPreservationGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "memory_detail_preservation_run1": GraderConfig(
                grader=grader,
            ),
            "memory_detail_preservation_run2": GraderConfig(
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
            grader_results=results["memory_detail_preservation_run1"],
            another_grader_results=results["memory_detail_preservation_run2"],
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
class TestMemoryDetailPreservationGraderAdversarial:
    """Adversarial tests for MemoryDetailPreservationGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with good and poor detail preservation pairs"""
        return [
            {
                "observation": "Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples and 3 green apples.",
                "good_memory": "Cabinet 1 at (3.5, 2.1): 5 red apples, 3 green apples.",
                "poor_memory": "Found some apples in a cabinet.",
                "task_context": "Task: Inventory items with precise locations and quantities",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Drawer 5 in Room B at position (10.2, 5.8) contains key-A123 with gold trim.",
                "good_memory": "Drawer 5, Room B, position (10.2, 5.8): key-A123, gold trim.",
                "poor_memory": "Found a key somewhere.",
                "task_context": "Task: Locate specific items with exact positions",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "On the shelf at height 1.5m, there are 12 red books, 8 blue notebooks, and 3 pencils.",
                "good_memory": "Shelf at 1.5m height: 12 red books, 8 blue notebooks, 3 pencils.",
                "poor_memory": "Found books and notebooks on a shelf.",
                "task_context": "Task: Count and record items accurately",
                "history_steps": [],
                "good_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Container 7 at location (15.3, 8.9, 2.4) stores 20 bolts, 15 nuts, and 10 washers.",
                "good_memory": "Container 7 at (15.3, 8.9, 2.4): 20 bolts, 15 nuts, 10 washers.",
                "poor_memory": "Container has hardware items.",
                "task_context": "Task: Precise inventory with 3D coordinates",
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
    async def test_adversarial_memory_detail_preservation_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = MemoryDetailPreservationGrader(model=model)

        # Configure GraderConfig to evaluate both good and poor memories
        grader_configs = {
            "memory_detail_preservation_good": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "memory": "good_memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "memory_detail_preservation_poor": GraderConfig(
                grader=grader,
                mapper={
                    "observation": "observation",
                    "memory": "poor_memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for poor memories
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_detail_preservation_poor"],
            label_path="poor_label",
        )

        # Use FalseNegativeAnalyzer for good memories
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_detail_preservation_good"],
            label_path="good_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.3, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
