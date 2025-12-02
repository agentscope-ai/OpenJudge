# -*- coding: utf-8 -*-
"""
Test Memory Accuracy Grader

Tests for the MemoryAccuracyGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import MemoryAccuracyGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_memory_accuracy_grader_creation():
    """Test creating a MemoryAccuracyGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="fake-api-key", stream=False)
    grader = MemoryAccuracyGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "memory_accuracy"


def test_memory_accuracy_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="fake-api-key", stream=False)
    grader = MemoryAccuracyGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_accuracy_poor():
    """Test detecting poor memory accuracy (hallucination)"""
    model = OpenAIChatModel(model="qwen3-32b", api_key="fake-api-key", stream=False)
    grader = MemoryAccuracyGrader(model=model)

    # Test case with inaccurate memory (hallucination)
    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        memory="There is a red vase inside the cabinet with gold trim.",
        task_context="Task: Inventory room objects",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor accuracy


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_accuracy_good():
    """Test with good memory accuracy"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = MemoryAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="Cabinet 1 contains 3 red apples.",
        memory="Cabinet 1 has 3 red apples.",
        task_context="Task: Inventory items",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good accuracy


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_accuracy_with_history():
    """Test memory accuracy with history"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = MemoryAccuracyGrader(model=model)

    history = [
        {"observation": "Cabinet is locked", "memory": "Cabinet 1 is locked"},
    ]

    result = await grader.aevaluate(
        observation="Cabinet is still locked. Cannot see inside.",
        memory="Cabinet contains 5 golden coins.",  # Cannot know this - poor accuracy
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
