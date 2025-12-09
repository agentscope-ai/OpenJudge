# -*- coding: utf-8 -*-
"""
Tests for RelevanceEvaluator
"""

from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.common.relevance import RelevanceEvaluator
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


@pytest.mark.asyncio
async def test_relevance_evaluator_basic():
    """Test RelevanceEvaluator with basic query and response"""
    # Initialize evaluator with mock model
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 5.0,
        "reason": "Relevance evaluation score: Response directly addresses the query",
    }
    model.achat = AsyncMock(return_value=mock_response)

    evaluator = RelevanceEvaluator(
        model=model,
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation
    result = await evaluator.aevaluate(
        query="What are Python decorators?",
        response="Decorators are functions that modify other functions. They use @syntax...",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert "Relevance evaluation score" in result.reason
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_with_context():
    """Test RelevanceEvaluator with context"""
    # Initialize evaluator with mock model
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 5.0,
        "reason": "Relevance evaluation score: Response is relevant with context",
    }
    model.achat = AsyncMock(return_value=mock_response)

    evaluator = RelevanceEvaluator(
        model=model,
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation with context
    result = await evaluator.aevaluate(
        query="What's the weather like then?",
        response="July is summer in Europe with warm weather between 20-25°C.",
        context="Previous conversation about planning a July vacation to Europe",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_chinese():
    """Test RelevanceEvaluator with Chinese language"""
    # Initialize evaluator with Chinese language
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 5.0,
        "reason": "相关性评估得分：回应与查询相关",
    }
    model.achat = AsyncMock(return_value=mock_response)

    evaluator = RelevanceEvaluator(
        model=model,
        threshold=0.7,
        language=LanguageEnum.ZH,
    )

    # Test evaluation
    result = await evaluator.aevaluate(
        query="什么是机器学习？",
        response="机器学习是人工智能的一个子集，它使系统能够从数据中学习。",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_with_ground_truth():
    """Test RelevanceEvaluator with ground truth for comparison"""
    # Initialize evaluator
    model = OpenAIChatModel(
        model="qwen-max",
        api_key="mock-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Mock the model response
    mock_response = AsyncMock()
    mock_response.metadata = {
        "score": 5.0,
        "reason": "Relevance evaluation score: Response is highly relevant",
    }
    model.achat = AsyncMock(return_value=mock_response)

    evaluator = RelevanceEvaluator(
        model=model,
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation with ground truth
    result = await evaluator.aevaluate(
        query="What services does the premium membership include?",
        response="The premium membership includes priority customer support, exclusive content access, and early product releases.",
        ground_truth="Premium membership: priority support, exclusive content, early releases, and monthly webinars.",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7
