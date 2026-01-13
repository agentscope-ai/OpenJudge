#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for MemoryRetrievalEffectivenessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using MemoryRetrievalEffectivenessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_retrieval_effectiveness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/memory/test_memory_retrieval_effectiveness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/memory/test_memory_retrieval_effectiveness.py -m quality
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
from openjudge.graders.agent import MemoryRetrievalEffectivenessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestMemoryRetrievalEffectivenessGraderUnit:
    """Unit tests for MemoryRetrievalEffectivenessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = MemoryRetrievalEffectivenessGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "memory_retrieval_effectiveness"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_effective_retrieval(self):
        """Test successful evaluation with effective memory retrieval"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Plan effectively uses memory information to guide actions",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryRetrievalEffectivenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will use the key I found earlier to unlock the door.",
                observation="You are near the locked door. Key is in inventory.",
                memory="Key-A was found in drawer 1.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "effective" in result.reason.lower() or "use" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_poor_retrieval(self):
        """Test evaluation detecting poor memory retrieval effectiveness"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Plan ignores memory information, proposing redundant search",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryRetrievalEffectivenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will search for the key in drawer 1.",
                observation="You are in the room.",
                memory="The key was already found in drawer 1 in step 3. Key is in inventory.",
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "ignore" in result.reason.lower() or "redundant" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.85,  # Will be normalized to 1.0
            "reason": "Plan effectively leverages memory given the task context",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = MemoryRetrievalEffectivenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "Found key in drawer 1", "memory": "Key in drawer 1"},
                {"step": 2, "observation": "Picked up key", "memory": "Key in inventory"},
            ]

            result = await grader.aevaluate(
                plan="I will use the key in my inventory to unlock the door.",
                observation="Standing near the locked door.",
                memory="Key is in inventory.",
                task_context="Task: Unlock the door",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "effective" in result.reason.lower() or "leverage" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = MemoryRetrievalEffectivenessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="Search the room.",
                observation="You are in a room.",
                memory="Room contains items.",
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
class TestMemoryRetrievalEffectivenessGraderQuality:
    """Quality tests for MemoryRetrievalEffectivenessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real memory retrieval failure cases"""
        return [
            # Case 1: Effective retrieval - plan uses memory appropriately
            {
                "plan": "I will use the key I found earlier to unlock the door.",
                "observation": "You are near the locked door. Key is in inventory.",
                "memory": "Key-A was found in drawer 1.",
                "task_context": "Task: Unlock the door",
                "history_steps": [],
                "human_score": 1,  # Effective retrieval
            },
            # Case 2: Poor retrieval - plan ignores memory (redundant search)
            {
                "plan": "I will search for the key in drawer 1.",
                "observation": "You are in the room.",
                "memory": "The key was already found in drawer 1 in step 3. Key is in inventory.",
                "task_context": "Task: Use the key to unlock the door",
                "history_steps": [],
                "human_score": 0,  # Poor retrieval effectiveness
            },
            # Case 3: Effective retrieval - uses location memory
            {
                "plan": "I will go to Cabinet 3 where the red book is located.",
                "observation": "You are in the library.",
                "memory": "Red book is in Cabinet 3 at position (5.2, 3.1).",
                "task_context": "Task: Retrieve the red book",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 4: Poor retrieval - searches already-checked location
            {
                "plan": "I will check Cabinet 1 for the red book.",
                "observation": "You are in the library.",
                "memory": "Cabinet 1 was checked and is empty. Red book is in Cabinet 3.",
                "task_context": "Task: Retrieve the red book",
                "history_steps": [],
                "human_score": 0,
            },
            # Case 5: Effective retrieval with multi-step memory
            {
                "plan": "I will take the opened path through the green door to the garden.",
                "observation": "Standing at the corridor intersection.",
                "memory": "Green door was unlocked in step 5. Blue door is still locked.",
                "task_context": "Task: Reach the garden efficiently",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 6: Poor retrieval - attempts locked path
            {
                "plan": "I will try to go through the blue door.",
                "observation": "Standing at the corridor intersection.",
                "memory": "Blue door is locked and no key was found. Green door was unlocked in step 5.",
                "task_context": "Task: Reach the garden efficiently",
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
        """Test the grader's ability to distinguish between effective and poor retrieval"""
        # Create grader with real model
        grader = MemoryRetrievalEffectivenessGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "memory_retrieval_effectiveness": GraderConfig(
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
            grader_results=results["memory_retrieval_effectiveness"],
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
        grader = MemoryRetrievalEffectivenessGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "memory_retrieval_effectiveness_run1": GraderConfig(
                grader=grader,
            ),
            "memory_retrieval_effectiveness_run2": GraderConfig(
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
            grader_results=results["memory_retrieval_effectiveness_run1"],
            another_grader_results=results["memory_retrieval_effectiveness_run2"],
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
class TestMemoryRetrievalEffectivenessGraderAdversarial:
    """Adversarial tests for MemoryRetrievalEffectivenessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with effective and poor retrieval pairs"""
        return [
            {
                "observation": "You are near the locked door. Key is in inventory.",
                "memory": "Key-A was found in drawer 1.",
                "effective_plan": "I will use the key I found earlier to unlock the door.",
                "poor_plan": "I will search for the key in drawer 1.",
                "task_context": "Task: Unlock the door",
                "history_steps": [],
                "effective_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "You are in the library.",
                "memory": "Cabinet 1 was checked and is empty. Red book is in Cabinet 3.",
                "effective_plan": "I will go to Cabinet 3 to get the red book.",
                "poor_plan": "I will check Cabinet 1 for the red book.",
                "task_context": "Task: Retrieve the red book",
                "history_steps": [],
                "effective_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "Standing at the corridor intersection.",
                "memory": "Blue door is locked and no key was found. Green door was unlocked in step 5.",
                "effective_plan": "I will take the opened path through the green door to the garden.",
                "poor_plan": "I will try to go through the blue door.",
                "task_context": "Task: Reach the garden efficiently",
                "history_steps": [
                    {"step": 5, "observation": "Unlocked green door", "memory": "Green door unlocked"},
                ],
                "effective_label": 1,
                "poor_label": 0,
            },
            {
                "observation": "You are in the workshop.",
                "memory": "Toolbox A contains hammer. Toolbox B contains wrench. Need wrench for current task.",
                "effective_plan": "I will get the wrench from Toolbox B.",
                "poor_plan": "I will search all toolboxes to find a wrench.",
                "task_context": "Task: Get the required tool quickly",
                "history_steps": [],
                "effective_label": 1,
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
    async def test_adversarial_memory_retrieval_effectiveness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = MemoryRetrievalEffectivenessGrader(model=model)

        # Configure GraderConfig to evaluate both effective and poor plans
        grader_configs = {
            "memory_retrieval_effectiveness_effective": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "effective_plan",
                    "observation": "observation",
                    "memory": "memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "memory_retrieval_effectiveness_poor": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "poor_plan",
                    "observation": "observation",
                    "memory": "memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for poor plans
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_retrieval_effectiveness_poor"],
            label_path="poor_label",
        )

        # Use FalseNegativeAnalyzer for effective plans
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["memory_retrieval_effectiveness_effective"],
            label_path="effective_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
