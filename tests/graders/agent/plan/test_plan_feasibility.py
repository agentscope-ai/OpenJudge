#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for PlanFeasibilityGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using PlanFeasibilityGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/plan/test_plan_feasibility.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/plan/test_plan_feasibility.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/plan/test_plan_feasibility.py -m quality
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
from openjudge.graders.agent import PlanFeasibilityGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestPlanFeasibilityGraderUnit:
    """Unit tests for PlanFeasibilityGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = PlanFeasibilityGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "plan_feasibility"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_feasible_plan(self):
        """Test successful evaluation with feasible plan"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Plan is feasible given current observation and state",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanFeasibilityGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will first open the drawer to get the key, then unlock the door.",
                observation="Drawer is closed. Key is inside.",
                memory="Key is in drawer 1.",
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "feasible" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_infeasible_plan(self):
        """Test evaluation detecting infeasible plan (impossible action)"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Plan is infeasible - attempts to use key that is not available yet",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanFeasibilityGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                plan="I will use the key to unlock the door.",
                observation="The drawer is closed. You don't have any items.",
                memory="The key is inside the drawer, but the drawer is not opened yet.",
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "infeasible" in result.reason.lower() or "not available" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_context_and_history(self):
        """Test evaluation with task context and history"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.85,  # Will be normalized to 1.0
            "reason": "Plan is feasible considering the task context and history",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanFeasibilityGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            history_steps = [
                {"step": 1, "observation": "Door is locked", "plan": "Need to find key"},
                {"step": 2, "observation": "Found drawer with key", "plan": "Will open drawer"},
            ]

            result = await grader.aevaluate(
                plan="I will take the key from the open drawer.",
                observation="Drawer is open. Key is visible inside.",
                memory="Key is in drawer 1, drawer is now open.",
                task_context="Task: Get the key to unlock the door",
                history_steps=history_steps,
            )

            # Assertions
            assert result.score == 1.0
            assert "feasible" in result.reason.lower()


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestPlanFeasibilityGraderQuality:
    """Quality tests for PlanFeasibilityGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from real plan impossible action cases"""
        return [
            # Case 1: Feasible plan - proper order of actions
            {
                "plan": "I will first open the drawer to get the key, then unlock the door.",
                "observation": "Drawer is closed. Key is inside.",
                "memory": "Key is in drawer 1.",
                "task_context": "Task: Unlock the door to exit",
                "history_steps": [],
                "human_score": 1,  # Feasible plan
            },
            # Case 2: Infeasible plan - using object before obtaining it
            {
                "plan": "I will use the key to unlock the door.",
                "observation": "The drawer is closed. You don't have any items.",
                "memory": "The key is inside the drawer, but the drawer is not opened yet.",
                "task_context": "Task: Unlock the door to exit",
                "history_steps": [],
                "human_score": 0,  # Infeasible plan
            },
            # Case 3: Feasible plan - using item in inventory
            {
                "plan": "I will use the key in my inventory to unlock the door.",
                "observation": "You are near the locked door. Key is in inventory.",
                "memory": "Key was obtained from drawer 1.",
                "task_context": "Task: Unlock the door",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 4: Infeasible plan - using non-existent item
            {
                "plan": "I will use the hammer to break the lock.",
                "observation": "You are near the locked door. Inventory is empty.",
                "memory": "No hammer was found in the area.",
                "task_context": "Task: Unlock the door",
                "history_steps": [],
                "human_score": 0,
            },
            # Case 5: Feasible plan - realistic action sequence
            {
                "plan": "I will examine the cabinet to see what's inside.",
                "observation": "You see a closed cabinet in front of you.",
                "memory": "Cabinet 1 has not been checked yet.",
                "task_context": "Task: Inventory room items",
                "history_steps": [],
                "human_score": 1,
            },
            # Case 6: Infeasible plan - action on locked object
            {
                "plan": "I will open the cabinet and take items out.",
                "observation": "Cabinet is locked. You need a key.",
                "memory": "Cabinet 1 is locked. No key in inventory.",
                "task_context": "Task: Get items from cabinet",
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
        """Test the grader's ability to distinguish between feasible and infeasible plans"""
        # Create grader with real model
        grader = PlanFeasibilityGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "plan_feasibility": GraderConfig(
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
            grader_results=results["plan_feasibility"],
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
        grader = PlanFeasibilityGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "plan_feasibility_run1": GraderConfig(
                grader=grader,
            ),
            "plan_feasibility_run2": GraderConfig(
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
            grader_results=results["plan_feasibility_run1"],
            another_grader_results=results["plan_feasibility_run2"],
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
class TestPlanFeasibilityGraderAdversarial:
    """Adversarial tests for PlanFeasibilityGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with feasible and infeasible plan pairs"""
        return [
            {
                "observation": "The drawer is closed. You don't have any items.",
                "memory": "The key is inside the drawer, but the drawer is not opened yet.",
                "feasible_plan": "I will first open the drawer to get the key, then unlock the door.",
                "infeasible_plan": "I will use the key to unlock the door.",
                "task_context": "Task: Unlock the door to exit",
                "history_steps": [],
                "feasible_label": 1,
                "infeasible_label": 0,
            },
            {
                "observation": "You are near the locked door. Inventory is empty.",
                "memory": "No tools were found in the area. Door requires a key.",
                "feasible_plan": "I will search the room for a key to unlock the door.",
                "infeasible_plan": "I will use the hammer to break the lock.",
                "task_context": "Task: Unlock the door",
                "history_steps": [],
                "feasible_label": 1,
                "infeasible_label": 0,
            },
            {
                "observation": "Cabinet is locked. You need a key.",
                "memory": "Cabinet 1 is locked. No key in inventory.",
                "feasible_plan": "I will search for the cabinet key before trying to open it.",
                "infeasible_plan": "I will open the cabinet and take items out.",
                "task_context": "Task: Get items from cabinet",
                "history_steps": [],
                "feasible_label": 1,
                "infeasible_label": 0,
            },
            {
                "observation": "You see a high shelf. Items are visible on top.",
                "memory": "Shelf is 2 meters high. No ladder or stool available.",
                "feasible_plan": "I will look for a ladder or stool to reach the shelf.",
                "infeasible_plan": "I will reach up and grab the items from the high shelf.",
                "task_context": "Task: Get items from the shelf",
                "history_steps": [],
                "feasible_label": 1,
                "infeasible_label": 0,
            },
            {
                "observation": "The door is locked. Key is in your inventory.",
                "memory": "You obtained key-A from the drawer.",
                "feasible_plan": "I will use the key in my inventory to unlock the door.",
                "infeasible_plan": "I will search for the key to unlock the door.",
                "task_context": "Task: Unlock and open the door",
                "history_steps": [],
                "feasible_label": 1,
                "infeasible_label": 0,
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
    async def test_adversarial_plan_feasibility_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = PlanFeasibilityGrader(model=model)

        # Configure GraderConfig to evaluate both feasible and infeasible plans
        grader_configs = {
            "plan_feasibility_feasible": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "feasible_plan",
                    "observation": "observation",
                    "memory": "memory",
                    "task_context": "task_context",
                    "history_steps": "history_steps",
                },
            ),
            "plan_feasibility_infeasible": GraderConfig(
                grader=grader,
                mapper={
                    "plan": "infeasible_plan",
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

        # Use FalsePositiveAnalyzer for infeasible plans
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["plan_feasibility_infeasible"],
            label_path="infeasible_label",
        )

        # Use FalseNegativeAnalyzer for feasible plans
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["plan_feasibility_feasible"],
            label_path="feasible_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.3, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
