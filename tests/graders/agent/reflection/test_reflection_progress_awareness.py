# -*- coding: utf-8 -*-
"""
Test Reflection Progress Awareness Grader

Tests for the ReflectionProgressAwarenessGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import ReflectionProgressAwarenessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_reflection_progress_awareness_grader_creation():
    """Test creating a ReflectionProgressAwarenessGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionProgressAwarenessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_progress_awareness"


def test_reflection_progress_awareness_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionProgressAwarenessGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_awareness_poor():
    """Test detecting poor progress awareness (overestimation)"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionProgressAwarenessGrader(model=model)

    # Test case with poor awareness (overestimated progress)
    result = await grader.aevaluate(
        observation="Cabinet 1 is still empty. No items found.",
        reflection="Excellent progress! I'm making great headway toward finding the apples!",
        task_context="Task: Find apples in cabinets",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor awareness


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_awareness_good():
    """Test with good progress awareness"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionProgressAwarenessGrader(model=model)

    result = await grader.aevaluate(
        observation="Found 3 apples in cabinet 2.",
        reflection="Good progress! Found apples as required.",
        task_context="Task: Find apples",
    )

    assert result is not None
    assert result.score == 1.0


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_awareness_with_history():
    """Test progress awareness with history showing repeated failures"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionProgressAwarenessGrader(model=model)

    history = [
        {"observation": "Cabinet 1 empty", "reflection": "No items"},
        {"observation": "Cabinet 1 still empty", "reflection": "Still nothing"},
        {"observation": "Cabinet 1 empty again", "reflection": "No progress"},
    ]

    result = await grader.aevaluate(
        observation="Cabinet 1 is empty.",
        reflection="Making excellent progress!",  # Poor awareness - overestimating
        history_steps=history,
        task_context="Task: Find items",
    )

    assert result is not None
    assert result.score == 0.0
