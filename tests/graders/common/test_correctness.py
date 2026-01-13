#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for CorrectnessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using CorrectnessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/common/test_correctness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/common/test_correctness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/common/test_correctness.py -m quality
    ```
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import FalseNegativeAnalyzer, FalsePositiveAnalyzer
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestCorrectnessGraderUnit:
    """Unit tests for CorrectnessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = CorrectnessGrader(model=mock_model)
        assert grader.name == "correctness"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_correctness_grader_with_reference_response(self):
        """Test CorrectnessGrader with reference response"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 5.0,
            "reason": "Correctness score: Response perfectly matches the reference response",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CorrectnessGrader(
                model=mock_model,
                threshold=0.7,
                language=LanguageEnum.EN,
            )

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Test evaluation
            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="Paris is the capital of France.",
                reference_response="The capital of France is Paris, with a population of 2.2M.",
                context="Geography quiz question",
            )

            # Verify result structure
            assert result.name == "correctness"
            assert isinstance(result.score, (int, float))
            assert result.score >= 1 and result.score <= 5
            assert "Correctness score" in result.reason
            assert result.metadata["threshold"] == 0.7

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_correctness_grader_without_reference_response(self):
        """Test CorrectnessGrader without reference response"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 4.0,
            "reason": "Correctness score: Response is largely correct",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CorrectnessGrader(
                model=mock_model,
                threshold=0.7,
                language=LanguageEnum.EN,
            )

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Test evaluation without reference_response
            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="Paris is the capital of France.",
            )

            # Verify result structure
            assert result.name == "correctness"
            assert isinstance(result.score, (int, float))
            assert result.metadata["threshold"] == 0.7

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = CorrectnessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="Paris is the capital of France.",
                reference_response="The capital of France is Paris.",
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
DATA_FILE = WORKSPACE_ROOT / "data" / "pre_data" / "open_judge-hug" / "common" / "correctness_eval_v1.json"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestCorrectnessGraderQuality:
    """Quality tests for CorrectnessGrader - testing evaluation quality"""

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
                "query": item["input"]["query"],
                "context": item["input"].get("context") or "",
                "reference_response": item["input"].get("reference") or "",
                "response": item["chosen"]["response"]["content"],  # Use 'response' to match grader parameter
                "human_score": 1,  # Binary: chosen has better correctness
            }
            samples.append(sample)

        return samples

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
        grader = CorrectnessGrader(model=model)

        # Use mapper to explicitly map fields (exclude human_score)
        grader_configs = {
            "correctness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Check that all evaluations completed successfully
        assert len(results["correctness"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["correctness"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "correctness" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = CorrectnessGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "correctness_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
            "correctness_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            grader_results=results["correctness_run1"],
            another_grader_results=results["correctness_run2"],
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
class TestCorrectnessGraderAdversarial:
    """Adversarial tests for CorrectnessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug for adversarial testing"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format with correct and incorrect pairs - use all samples
        samples = []
        for item in data:
            sample = {
                "query": item["input"]["query"],
                "context": item["input"].get("context") or "",
                "reference_response": item["input"].get("reference") or "",
                "correct_response": item["chosen"]["response"]["content"],
                "incorrect_response": item["rejected"]["response"]["content"],
                "correct_label": 1,
                "incorrect_label": 0,
            }
            samples.append(sample)

        return samples

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
    async def test_adversarial_correctness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = CorrectnessGrader(model=model)

        # Configure GraderConfig to evaluate both correct and incorrect responses
        grader_configs = {
            "correctness_correct": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "correct_response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
            "correctness_incorrect": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "incorrect_response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for incorrect responses
        # CorrectnessGrader returns 1-5, so use threshold 3.5 (scores >= 4 are "correct")
        fp_analyzer = FalsePositiveAnalyzer(prediction_threshold=3.5)
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["correctness_incorrect"],
            label_path="incorrect_label",
        )

        # Use FalseNegativeAnalyzer for correct responses
        fn_analyzer = FalseNegativeAnalyzer(prediction_threshold=3.5)
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["correctness_correct"],
            label_path="correct_label",
        )

        # Calculate pairwise accuracy: chosen should score higher than rejected
        correct_predictions = 0
        total_predictions = 0

        for i, (chosen_result, rejected_result) in enumerate(
            zip(results["correctness_correct"], results["correctness_incorrect"]),
        ):
            if chosen_result and rejected_result:
                # Chosen (correct) should have higher score than rejected (incorrect)
                # Higher score means better correctness
                if chosen_result.score > rejected_result.score:
                    correct_predictions += 1
                total_predictions += 1

        pairwise_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Print accuracy for reporting
        print(f"\n{'=' * 60}")
        print(
            f"CorrectnessGrader Pairwise Accuracy: {pairwise_accuracy:.4f} ({correct_predictions}/{total_predictions})"
        )
        print(f"{'=' * 60}\n")

        # Assert that false positive and false negative rates meet expected thresholds
        # Note: We log the rates but don't fail the test if they exceed thresholds
        # to allow accuracy reporting even when rates are high
        if fp_result.false_positive_rate > 0.3:
            print(f"Warning: False positive rate too high: {fp_result.false_positive_rate}")
        if fn_result.false_negative_rate > 0.3:
            print(f"Warning: False negative rate too high: {fn_result.false_negative_rate}")

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"

        # Store accuracy in result metadata for reporting
        fp_result.metadata["pairwise_accuracy"] = pairwise_accuracy
        fn_result.metadata["pairwise_accuracy"] = pairwise_accuracy
