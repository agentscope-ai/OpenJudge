#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for TextToImageGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using TextToImageGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/multimodal/test_text_to_image.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/multimodal/test_text_to_image.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/multimodal/test_text_to_image.py -m quality
    ```
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import FalseNegativeAnalyzer, FalsePositiveAnalyzer
from openjudge.graders.multimodal._internal import MLLMImage
from openjudge.graders.multimodal.text_to_image import TextToImageGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestTextToImageGraderUnit:
    """Unit tests for TextToImageGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        # Create a mock that properly inherits from BaseChatModel
        mock_model = AsyncMock(spec=BaseChatModel)
        grader = TextToImageGrader(model=mock_model)
        assert grader.name == "text_to_image"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""

        # Create simple mock response objects (not AsyncMock to avoid __aiter__ check)
        class MockResponse:
            def __init__(self, score, reason):
                self.parsed = {"score": score, "reason": reason}

        # TextToImageGrader calls model twice (semantic + perceptual)
        # Scores are now in 1-5 range
        mock_semantic = MockResponse(4.0, "Good semantic consistency")
        mock_perceptual = MockResponse(4.0, "Good perceptual quality")

        # Create mock model
        mock_model = AsyncMock(spec=BaseChatModel)
        # Return different responses for semantic and perceptual calls
        mock_model.achat = AsyncMock(side_effect=[mock_semantic, mock_perceptual])

        grader = TextToImageGrader(model=mock_model)

        # Create mock image with online URL
        mock_image = MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        result = await grader.aevaluate(
            query="A cat sitting on a blue sofa",
            response=mock_image,
        )

        # Assertions - score is geometric mean sqrt(4*4) = 4.0 (scores in 1-5 range)
        assert 3.5 <= result.score <= 4.5  # Allow some tolerance
        assert len(result.reason) > 0  # Has a reason

        # Verify model was called twice (semantic + perceptual)
        assert mock_model.achat.call_count == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        from openjudge.graders.base_grader import GraderError

        # Create mock model that raises exception
        mock_model = AsyncMock(spec=BaseChatModel)
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = TextToImageGrader(model=mock_model)

        # Create mock image with online URL
        mock_image = MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        result = await grader.aevaluate(
            query="A dog in a park",
            response=mock_image,
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
DATA_FILE = WORKSPACE_ROOT / "data" / "pre_data" / "open_judge-hug" / "multimodal" / "text_to_image_eval_v1.json"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestTextToImageGraderQuality:
    """Quality tests for TextToImageGrader - testing evaluation quality"""

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
                "chosen_model": item["chosen"]["response"]["model"],
                "rejected_model": item["rejected"]["response"]["model"],
                "human_score": 1,  # Binary: image matches query well
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
        grader = TextToImageGrader(model=model)

        # Custom mapper to construct query and image response
        def map_text_to_image(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")

            # Create MLLMImage from base64
            mllm_image = MLLMImage(base64=image_base64, format="png")

            return {
                "query": query,
                "response": mllm_image,
            }

        # Use custom mapper
        grader_configs = {
            "text_to_image": GraderConfig(
                grader=grader,
                mapper=map_text_to_image,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Check that all evaluations completed successfully
        assert len(results["text_to_image"]) == len(dataset)

        # Check that scores are in valid range (1-5 for text_to_image)
        for result in results["text_to_image"]:
            assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

        # Verify analysis results structure
        assert "text_to_image" in results

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = TextToImageGrader(model=model)

        # Custom mapper to construct query and image response
        def map_text_to_image(sample):
            query = sample.get("query", "")
            image_base64 = sample.get("image_base64", "")

            # Create MLLMImage from base64
            mllm_image = MLLMImage(base64=image_base64, format="png")

            return {
                "query": query,
                "response": mllm_image,
            }

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "text_to_image_run1": GraderConfig(
                grader=grader,
                mapper=map_text_to_image,
            ),
            "text_to_image_run2": GraderConfig(
                grader=grader,
                mapper=map_text_to_image,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            grader_results=results["text_to_image_run1"],
            another_grader_results=results["text_to_image_run2"],
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
class TestTextToImageGraderAdversarial:
    """Adversarial tests for TextToImageGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load evaluation dataset from openjudge-hug for adversarial testing"""
        import json

        if not DATA_FILE.exists():
            pytest.skip(f"Data file not found: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Transform to test format with chosen (better) and rejected (worse) pairs
        samples = []
        for item in data:
            # Extract chosen image
            chosen_media = item.get("chosen", {}).get("response", {}).get("media_contents", [])
            chosen_image_data = None
            if chosen_media:
                for media in chosen_media:
                    if media["type"] == "image":
                        chosen_image_data = media["content"]["image"]["data"]
                        break

            # Extract rejected image
            rejected_media = item.get("rejected", {}).get("response", {}).get("media_contents", [])
            rejected_image_data = None
            if rejected_media:
                for media in rejected_media:
                    if media["type"] == "image":
                        rejected_image_data = media["content"]["image"]["data"]
                        break

            # Only include samples that have both images
            if chosen_image_data and rejected_image_data:
                sample = {
                    "query": item["input"]["query"],
                    "chosen_image_base64": chosen_image_data,
                    "rejected_image_base64": rejected_image_data,
                    "chosen_label": 1,
                    "rejected_label": 0,
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
    async def test_adversarial_text_to_image_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = TextToImageGrader(model=model)

        # Custom mappers for chosen and rejected images
        def map_chosen_image(sample):
            query = sample.get("query", "")
            chosen_image_base64 = sample.get("chosen_image_base64", "")
            mllm_image = MLLMImage(base64=chosen_image_base64, format="png")
            return {"query": query, "response": mllm_image}

        def map_rejected_image(sample):
            query = sample.get("query", "")
            rejected_image_base64 = sample.get("rejected_image_base64", "")
            mllm_image = MLLMImage(base64=rejected_image_base64, format="png")
            return {"query": query, "response": mllm_image}

        # Configure GraderConfig to evaluate both chosen and rejected images
        grader_configs = {
            "text_to_image_chosen": GraderConfig(
                grader=grader,
                mapper=map_chosen_image,
            ),
            "text_to_image_rejected": GraderConfig(
                grader=grader,
                mapper=map_rejected_image,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for rejected images
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["text_to_image_rejected"],
            label_path="rejected_label",
        )

        # Use FalseNegativeAnalyzer for chosen images
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["text_to_image_chosen"],
            label_path="chosen_label",
        )

        # Calculate pairwise accuracy: chosen should score higher than rejected
        correct_predictions = 0
        total_predictions = 0

        for i, (chosen_result, rejected_result) in enumerate(
            zip(results["text_to_image_chosen"], results["text_to_image_rejected"]),
        ):
            if chosen_result and rejected_result:
                # Chosen should have higher score than rejected
                if chosen_result.score > rejected_result.score:
                    correct_predictions += 1
                total_predictions += 1

        pairwise_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Print accuracy for reporting
        print(f"\n{'=' * 60}")
        print(
            f"TextToImageGrader Pairwise Accuracy: {pairwise_accuracy:.4f} ({correct_predictions}/{total_predictions})"
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
