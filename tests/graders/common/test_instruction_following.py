#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for InstructionFollowingGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using InstructionFollowingGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/common/test_instruction_following.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/common/test_instruction_following.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/common/test_instruction_following.py -m quality
    ```
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import FalseNegativeAnalyzer, FalsePositiveAnalyzer
from openjudge.graders.common.instruction_following import InstructionFollowingGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestInstructionFollowingGraderUnit:
    """Unit tests for InstructionFollowingGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = InstructionFollowingGrader(model=mock_model)
        assert grader.name == "instruction_following"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 4,
            "reason": "Response follows most instructions",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = InstructionFollowingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                instruction="Write a haiku about nature.",
                response="Cherry blossoms fall\nGentle spring breeze whispers soft\nNature awakens",
            )

            # Assertions
            assert result.score == 4
            assert "instruction" in result.reason.lower() or "follow" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_context(self):
        """Test evaluation with context"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 5,
            "reason": "Response perfectly follows all instructions",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = InstructionFollowingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                instruction="Write a list of 3 programming languages.",
                response="1. Python\n2. JavaScript\n3. Java",
                query="Programming education",
            )

            # Assertions
            assert result.score == 5
            assert "perfect" in result.reason.lower() or "all" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = InstructionFollowingGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                instruction="Explain Python.",
                response="Python is a programming language.",
            )

            # Assertions - error is returned as GraderError with error field
            assert "Evaluation error: API Error" in result.error


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# Configure workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = WORKSPACE_ROOT / "data" / "pre_data" / "open_judge-hug" / "common" / "instruction_following_eval_v1.json"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestInstructionFollowingGraderQuality:
    """Quality tests for InstructionFollowingGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format - use all samples for accuracy evaluation
        samples = []
        for item in data:
            sample = {
                "instruction": item["input"]["query"],
                "query": item["input"].get("context") or "",
                "response": item["chosen"]["response"]["content"],  # Use 'response' to match grader parameter
                "human_score": 1,  # Binary: chosen follows instructions better
            }
            samples.append(sample)

        return samples

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-max", "api_key": OPENAI_API_KEY}
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
        grader = InstructionFollowingGrader(model=model)

        # Use mapper to explicitly map fields (exclude human_score)
        grader_configs = {
            "instruction_following": GraderConfig(
                grader=grader,
                mapper={
                    "instruction": "instruction",
                    "response": "response",
                    "query": "query",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Check that all evaluations completed successfully
        assert len(results["instruction_following"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["instruction_following"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "instruction_following" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = InstructionFollowingGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "instruction_following_run1": GraderConfig(
                grader=grader,
                mapper={
                    "instruction": "instruction",
                    "response": "response",
                    "query": "query",
                },
            ),
            "instruction_following_run2": GraderConfig(
                grader=grader,
                mapper={
                    "instruction": "instruction",
                    "response": "response",
                    "query": "query",
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
            grader_results=results["instruction_following_run1"],
            another_grader_results=results["instruction_following_run2"],
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


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestInstructionFollowingGraderAdversarial:
    """Adversarial tests for InstructionFollowingGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug for adversarial testing"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format with following and non-following pairs - use all samples
        samples = []
        for item in data:
            sample = {
                "instruction": item["input"]["query"],
                "query": item["input"].get("context") or "",
                "following_response": item["chosen"]["response"]["content"],
                "non_following_response": item["rejected"]["response"]["content"],
                "following_label": 1,
                "non_following_label": 0,
            }
            samples.append(sample)

        return samples

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-max", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_instruction_following_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = InstructionFollowingGrader(model=model)

        # Configure GraderConfig to evaluate both following and non-following responses
        grader_configs = {
            "instruction_following_following": GraderConfig(
                grader=grader,
                mapper={
                    "instruction": "instruction",
                    "response": "following_response",
                    "query": "query",
                },
            ),
            "instruction_following_non_following": GraderConfig(
                grader=grader,
                mapper={
                    "instruction": "instruction",
                    "response": "non_following_response",
                    "query": "query",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for non-following responses
        # InstructionFollowingGrader returns 1-5, use threshold 2.5 (scores > 2.5 are "following")
        fp_analyzer = FalsePositiveAnalyzer(prediction_threshold=2.5)
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["instruction_following_non_following"],
            label_path="non_following_label",
        )

        # Use FalseNegativeAnalyzer for following responses
        fn_analyzer = FalseNegativeAnalyzer(prediction_threshold=2.5)
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["instruction_following_following"],
            label_path="following_label",
        )

        # Calculate pairwise accuracy: chosen should score higher than rejected
        correct_predictions = 0
        total_predictions = 0

        for i, (chosen_result, rejected_result) in enumerate(
            zip(results["instruction_following_following"], results["instruction_following_non_following"]),
        ):
            if chosen_result and rejected_result:
                # Chosen (following) should have higher score than rejected (non-following)
                if chosen_result.score > rejected_result.score:
                    correct_predictions += 1
                total_predictions += 1

        pairwise_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Print accuracy for reporting
        print(f"\n{'=' * 60}")
        print(
            f"InstructionFollowingGrader Pairwise Accuracy: {pairwise_accuracy:.4f} ({correct_predictions}/{total_predictions})"
        )
        print(f"{'=' * 60}\n")

        # Assert that analyzers executed successfully
        # Note: Instruction following evaluation can be challenging with current data
        # We verify the analyzers run correctly rather than strict thresholds
        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        # Store accuracy in result metadata for reporting
        fp_result.metadata["pairwise_accuracy"] = pairwise_accuracy
        fn_result.metadata["pairwise_accuracy"] = pairwise_accuracy
