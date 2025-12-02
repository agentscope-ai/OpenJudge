# -*- coding: utf-8 -*-
"""
Test Reflection Outcome Understanding Grader

Tests for the ReflectionOutcomeUnderstandingGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import (
    ReflectionOutcomeUnderstandingGrader,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_reflection_outcome_understanding_grader_creation():
    """Test creating a ReflectionOutcomeUnderstandingGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_outcome_understanding"


def test_reflection_outcome_understanding_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionOutcomeUnderstandingGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_understanding_poor():
    """Test detecting poor outcome understanding (misinterpretation)"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    # Test case with poor understanding (misinterpretation)
    result = await grader.aevaluate(
        observation="The drawer is still closed. Action failed.",
        reflection="I successfully opened the drawer.",
        task_context="Task: Open the drawer",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor understanding


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_understanding_good():
    """Test with good outcome understanding"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    # Test case with good understanding
    result = await grader.aevaluate(
        observation="The drawer is now open.",
        reflection="I successfully opened the drawer.",
        task_context="Task: Open the drawer",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good understanding


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_understanding_with_history():
    """Test outcome understanding with history"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    history = [
        {"observation": "Drawer is locked", "reflection": "Need to find key"},
    ]

    result = await grader.aevaluate(
        observation="Drawer is still locked.",
        reflection="The drawer is now open.",  # Poor understanding
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
