# -*- coding: utf-8 -*-
"""
Test Action Alignment Grader

Tests for the ActionAlignmentGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import ActionAlignmentGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_action_alignment_grader_creation():
    """Test creating an ActionAlignmentGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ActionAlignmentGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "action_alignment"


def test_action_alignment_grader_creation_chinese():
    """Test creating a Chinese ActionAlignmentGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ActionAlignmentGrader(model=model, language=LanguageEnum.ZH)

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_alignment_poor():
    """Test detecting poor action alignment"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ActionAlignmentGrader(model=model)

    # Test case with poor action alignment (misalignment)
    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="close drawer 1",
        task_context="Task: Find the key in the room",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert result.score == 0.0  # Should detect poor alignment
    assert "alignment" in result.reason.lower() or "contradict" in result.reason.lower()


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_alignment_good():
    """Test with good action alignment"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ActionAlignmentGrader(model=model)

    # Test case with good alignment
    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="open drawer 1",
        task_context="Task: Find the key",
    )

    assert result is not None
    assert result.score == 1.0  # Should be good alignment


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_alignment_with_history():
    """Test action alignment with history steps"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ActionAlignmentGrader(model=model)

    history = [
        {"plan": "Check drawer 1", "action": "examine drawer 1"},
        {"plan": "Look inside", "action": "open drawer 1"},
    ]

    result = await grader.aevaluate(
        plan="I will close the drawer after searching.",
        action="open drawer 2",  # Wrong action
        history_steps=history,
        task_context="Task: Search for items",
    )

    assert result is not None
    assert hasattr(result, "score")
