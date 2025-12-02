# -*- coding: utf-8 -*-
"""
Test Reflection Accuracy Grader

Tests for the ReflectionAccuracyGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import ReflectionAccuracyGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_reflection_accuracy_grader_creation():
    """Test creating a ReflectionAccuracyGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionAccuracyGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_accuracy"


def test_reflection_accuracy_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionAccuracyGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_accuracy_poor():
    """Test detecting poor reflection accuracy (hallucination)"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionAccuracyGrader(model=model)

    # Test case with inaccurate reflection (hallucinated details)
    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        reflection="I observed a red vase on top of the cabinet with three flowers.",
        task_context="Task: Inventory room objects",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor accuracy


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_accuracy_good():
    """Test with good reflection accuracy"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet and a table.",
        reflection="I observed a closed cabinet and a table in the room.",
        task_context="Task: Inventory room",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good accuracy


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_accuracy_with_history():
    """Test reflection accuracy with history"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionAccuracyGrader(model=model)

    history = [
        {"observation": "Empty room", "reflection": "Room has no objects"},
    ]

    result = await grader.aevaluate(
        observation="You see an empty room.",
        reflection="I see a golden statue in the corner.",  # Inaccurate - hallucinated
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
