# -*- coding: utf-8 -*-
"""
Tests for CorrectnessGrader
"""

from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.common.correctness import CorrectnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


@pytest.mark.asyncio
async def test_correctness_grader_with_ground_truth():
    """Test CorrectnessGrader with ground truth"""
    # Initialize grader with mock model
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 5.0,
        "reason": "Correctness score: Response perfectly matches the ground truth",
    }
    model.achat = AsyncMock(return_value=mock_response)

    grader = CorrectnessGrader(
        model=model,
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        ground_truth="The capital of France is Paris, with a population of 2.2M.",
        context="Geography quiz question",
    )

    # Verify result structure
    assert result.name == "correctness"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert "Correctness score" in result.reason
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_correctness_grader_without_ground_truth():
    """Test CorrectnessGrader without ground truth"""
    # Initialize grader with mock model
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 4.0,
        "reason": "Correctness score: Response is largely correct",
    }
    model.achat = AsyncMock(return_value=mock_response)

    grader = CorrectnessGrader(
        model=model,
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation without ground_truth
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
    )

    # Verify result structure
    assert result.name == "correctness"
    assert isinstance(result.score, (int, float))
    assert result.metadata["threshold"] == 0.7
