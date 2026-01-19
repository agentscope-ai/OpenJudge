#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ImageHelpfulnessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ImageHelpfulnessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multimodal/test_image_helpfulness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multimodal/test_image_helpfulness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multimodal/test_image_helpfulness.py -m quality
    ```
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import FalseNegativeAnalyzer, FalsePositiveAnalyzer
from openjudge.graders.multimodal._internal import MLLMImage
from openjudge.graders.multimodal.image_helpfulness import ImageHelpfulnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestImageHelpfulnessGraderUnit:
    """Unit tests for ImageHelpfulnessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ImageHelpfulnessGrader(model=mock_model)
        assert grader.name == "image_helpfulness"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""

        # Create a simple mock response object (not AsyncMock to avoid __aiter__ check)
        class MockResponse:
            def __init__(self):
                self.parsed = {
                    "score": 4.0,  # Score in 1-5 range
                    "reason": "Image is very helpful for understanding the text",
                }

        mock_response = MockResponse()

        # Create mock model
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = ImageHelpfulnessGrader(model=mock_model)

        # Create mock image with online URL
        mock_image = MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        result = await grader.aevaluate(
            response=["The system architecture:", mock_image, "shows the component interactions"],
        )

        # Assertions
        assert result.score == 4.0  # Score in 1-5 range
        assert "helpful" in result.reason.lower()

        # Verify model was called correctly
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        from openjudge.graders.base_grader import GraderError

        # Create mock model that raises exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = ImageHelpfulnessGrader(model=mock_model)

        # Create mock image with online URL
        mock_image = MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        result = await grader.aevaluate(
            response=["Text before", mock_image, "Text after"],
        )

        # Assertions - grader returns GraderError on exception
        assert isinstance(result, GraderError)
        assert "Evaluation error: API Error" in result.error


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# Configure workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = WORKSPACE_ROOT / "data" / "pre_data" / "open_judge-hug" / "multimodal" / "image_helpfulness_eval_v1.json"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestImageHelpfulnessGraderQuality:
    """Quality tests for ImageHelpfulnessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format
        samples = []
        for item in data:
            # Extract image from media_contents
            media_contents = item["input"].get("media_contents", [])
            image_data = None
            if media_contents:
                for media in media_contents:
                    if media["type"] == "image":
                        image_data = media["content"]["image"]["data"]
                        break

            if not image_data:
                continue

            sample = {
                "query": item["input"]["query"],
                "image_base64": image_data,
                "chosen_response": item["chosen"]["response"]["content"],
                "rejected_response": item["rejected"]["response"]["content"],
                "chosen_model": item["chosen"]["response"]["model"],
                "rejected_model": item["rejected"]["response"]["model"],
                "human_score_chosen": 1,  # Chosen is more helpful
                "human_score_rejected": 0,  # Rejected is less helpful
            }
            samples.append(sample)

        return samples  # Use all samples for comprehensive testing

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-vl-max", "api_key": OPENAI_API_KEY}
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
        grader = ImageHelpfulnessGrader(model=model)

        # Custom mapper to construct response with image
        def map_response_with_image(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")
            chosen_response = sample.get("chosen_response", "")

            # Create MLLMImage from base64
            mllm_image = MLLMImage(base64=image_base64, format="png")

            # Construct multimodal response
            return {
                "response": [query, mllm_image, chosen_response],
            }

        # Use custom mapper
        grader_configs = {
            "image_helpfulness": GraderConfig(
                grader=grader,
                mapper=map_response_with_image,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Check that all evaluations completed successfully
        assert len(results["image_helpfulness"]) == len(dataset)

        # Check that scores are in valid range (1-5 for image helpfulness)
        for result in results["image_helpfulness"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "image_helpfulness" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ImageHelpfulnessGrader(model=model)

        # Custom mapper to construct response with image
        def map_response_with_image(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")
            chosen_response = sample.get("chosen_response", "")

            # Create MLLMImage from base64
            mllm_image = MLLMImage(base64=image_base64, format="png")

            # Construct multimodal response
            return {
                "response": [query, mllm_image, chosen_response],
            }

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "image_helpfulness_run1": GraderConfig(
                grader=grader,
                mapper=map_response_with_image,
            ),
            "image_helpfulness_run2": GraderConfig(
                grader=grader,
                mapper=map_response_with_image,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            grader_results=results["image_helpfulness_run1"],
            another_grader_results=results["image_helpfulness_run2"],
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
class TestImageHelpfulnessGraderAdversarial:
    """Adversarial tests for ImageHelpfulnessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug for adversarial testing"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format with helpful and unhelpful pairs
        samples = []
        for item in data:
            # Extract image from media_contents
            media_contents = item["input"].get("media_contents", [])
            image_data = None
            if media_contents:
                for media in media_contents:
                    if media["type"] == "image":
                        image_data = media["content"]["image"]["data"]
                        break

            if not image_data:
                continue

            sample = {
                "query": item["input"]["query"],
                "image_base64": image_data,
                "helpful_response": item["chosen"]["response"]["content"],
                "unhelpful_response": item["rejected"]["response"]["content"],
                "helpful_label": 1,
                "unhelpful_label": 0,
            }
            samples.append(sample)

        return samples  # Use all samples for comprehensive testing

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-vl-max", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_image_helpfulness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ImageHelpfulnessGrader(model=model)

        # Custom mappers for helpful and unhelpful responses
        def map_helpful_response(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")
            helpful_response = sample.get("helpful_response", "")
            mllm_image = MLLMImage(base64=image_base64, format="png")
            return {"response": [query, mllm_image, helpful_response]}

        def map_unhelpful_response(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")
            unhelpful_response = sample.get("unhelpful_response", "")
            mllm_image = MLLMImage(base64=image_base64, format="png")
            return {"response": [query, mllm_image, unhelpful_response]}

        # Configure GraderConfig to evaluate both helpful and unhelpful responses
        grader_configs = {
            "image_helpfulness_helpful": GraderConfig(
                grader=grader,
                mapper=map_helpful_response,
            ),
            "image_helpfulness_unhelpful": GraderConfig(
                grader=grader,
                mapper=map_unhelpful_response,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for unhelpful responses
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["image_helpfulness_unhelpful"],
            label_path="unhelpful_label",
        )

        # Use FalseNegativeAnalyzer for helpful responses
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["image_helpfulness_helpful"],
            label_path="helpful_label",
        )

        # Calculate pairwise accuracy: helpful should score higher than unhelpful
        correct_predictions = 0
        total_predictions = 0

        for i, (helpful_result, unhelpful_result) in enumerate(
            zip(results["image_helpfulness_helpful"], results["image_helpfulness_unhelpful"]),
        ):
            if helpful_result and unhelpful_result:
                # Helpful should have higher score than unhelpful
                if helpful_result.score > unhelpful_result.score:
                    correct_predictions += 1
                total_predictions += 1

        pairwise_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Print accuracy for reporting
        print(f"\n{'=' * 60}")
        print(
            f"ImageHelpfulnessGrader Pairwise Accuracy: {pairwise_accuracy:.4f} ({correct_predictions}/{total_predictions})"
        )
        print(f"{'=' * 60}\n")

        # Assert that analyzers executed successfully
        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        # Store accuracy in result metadata for reporting
        fp_result.metadata["pairwise_accuracy"] = pairwise_accuracy
        fn_result.metadata["pairwise_accuracy"] = pairwise_accuracy
