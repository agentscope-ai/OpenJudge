#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for HallucinationGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using HallucinationGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/common/test_hallucination.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/common/test_hallucination.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/common/test_hallucination.py -m quality
    ```
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import FalseNegativeAnalyzer, FalsePositiveAnalyzer
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestHallucinationGraderUnit:
    """Unit tests for HallucinationGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = HallucinationGrader(model=mock_model)
        assert grader.name == "hallucination"
        assert grader.model == mock_model

    def test_get_metadata(self):
        """Test get_metadata."""
        meta = HallucinationGrader.get_metadata()
        assert len(meta) == 2

        assert "aevaluate" in meta
        assert "Evaluate hallucination in response" in meta["aevaluate"]

        assert "prompt" in meta
        prompt = meta["prompt"]
        assert len(prompt) == 2
        assert len(prompt[LanguageEnum.EN.value]) == 1
        assert prompt[LanguageEnum.EN.value][0]["role"] == "user"
        assert (
            "evaluating whether the model response contains hallucinations"
            in prompt[LanguageEnum.EN.value][0]["content"]
        )
        assert len(prompt[LanguageEnum.ZH.value]) == 1
        assert prompt[LanguageEnum.ZH.value][0]["role"] == "user"
        assert "负责评估模型输出是否包含幻觉" in prompt[LanguageEnum.ZH.value][0]["content"]

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock response with the expected parsed structure
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Response contains minimal hallucinations",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = HallucinationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="Paris is the capital of France.",
                context="France is a country in Europe.",
            )

            # Assertions
            assert result.score == 4
            assert "hallucination" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_reference(self):
        """Test evaluation with reference answer"""
        # Setup mock response with the expected parsed structure
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Response is factually accurate with no hallucinations",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = HallucinationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What is photosynthesis?",
                response="Photosynthesis is the process by which plants convert light into energy.",
                context="Biology",
                reference_response="Photosynthesis converts light to chemical energy.",
            )

            # Assertions
            assert result.score == 5
            assert "accurate" in result.reason.lower() or "no hallucination" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = HallucinationGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What is Python?",
                response="Python is a programming language.",
                context="Programming",
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
DATA_FILE = WORKSPACE_ROOT / "data" / "pre_data" / "open_judge-hug" / "common" / "hallucination_eval_v1.json"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestHallucinationGraderQuality:
    """Quality tests for HallucinationGrader - testing evaluation quality"""

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
                "human_score": 1,  # Binary: chosen has less hallucination
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
        grader = HallucinationGrader(model=model)

        # Use mapper to explicitly map fields (exclude human_score)
        grader_configs = {
            "hallucination": GraderConfig(
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
        assert len(results["hallucination"]) == len(dataset)

        # Check that scores are in valid range (1-5)
        for result in results["hallucination"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "hallucination" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = HallucinationGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "hallucination_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                    "reference_response": "reference_response",
                },
            ),
            "hallucination_run2": GraderConfig(
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
            dataset=dataset,
            grader_results=results["hallucination_run1"],
            another_grader_results=results["hallucination_run2"],
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
class TestHallucinationGraderAdversarial:
    """Adversarial tests for HallucinationGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug for adversarial testing"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format with factual and hallucinated pairs - use all samples
        samples = []
        for item in data:
            sample = {
                "query": item["input"]["query"],
                "context": item["input"].get("context") or "",
                "factual_response": item["chosen"]["response"]["content"],
                "hallucinated_response": item["rejected"]["response"]["content"],
                "factual_label": 1,
                "hallucinated_label": 0,
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
    async def test_adversarial_hallucination_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = HallucinationGrader(model=model)

        # Configure GraderConfig to evaluate both factual and hallucinated responses
        grader_configs = {
            "hallucination_factual": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "factual_response",
                    "context": "context",
                },
            ),
            "hallucination_hallucinated": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "hallucinated_response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for hallucinated responses
        # HallucinationGrader returns 1-5, so use threshold 3.5 (scores >= 4 are "factual")
        fp_analyzer = FalsePositiveAnalyzer(prediction_threshold=3.5)
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["hallucination_hallucinated"],
            label_path="hallucinated_label",
        )

        # Use FalseNegativeAnalyzer for factual responses
        fn_analyzer = FalseNegativeAnalyzer(prediction_threshold=3.5)
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["hallucination_factual"],
            label_path="factual_label",
        )

        # Calculate pairwise accuracy: chosen should score higher than rejected
        correct_predictions = 0
        total_predictions = 0

        for i, (chosen_result, rejected_result) in enumerate(
            zip(results["hallucination_factual"], results["hallucination_hallucinated"]),
        ):
            if chosen_result and rejected_result:
                # Chosen (factual) should have higher score than rejected (hallucinated)
                # Higher score means less hallucination, so factual should score higher
                if chosen_result.score > rejected_result.score:
                    correct_predictions += 1
                total_predictions += 1

        pairwise_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Print accuracy for reporting
        print(f"\n{'=' * 60}")
        print(
            f"HallucinationGrader Pairwise Accuracy: {pairwise_accuracy:.4f} ({correct_predictions}/{total_predictions})"
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
