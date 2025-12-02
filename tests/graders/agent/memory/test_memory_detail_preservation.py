# -*- coding: utf-8 -*-
"""
Test Memory Detail Preservation Grader

Tests for the MemoryDetailPreservationGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import MemoryDetailPreservationGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_memory_detail_preservation_grader_creation():
    """Test creating a MemoryDetailPreservationGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="fake-api-key", stream=False)
    grader = MemoryDetailPreservationGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "memory_detail_preservation"


def test_memory_detail_preservation_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="fake-api-key", stream=False)
    grader = MemoryDetailPreservationGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_detail_preservation_poor():
    """Test detecting poor detail preservation (over-simplification)"""
    model = OpenAIChatModel(model="qwen3-32b", api_key="fake-api-key", stream=False)
    grader = MemoryDetailPreservationGrader(model=model)

    # Test case with poor detail preservation (lost important details)
    result = await grader.aevaluate(
        observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples and 3 green apples.",
        memory="Found some apples in a cabinet.",  # Too vague
        task_context="Task: Inventory items with precise locations and quantities",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor detail preservation


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_detail_preservation_good():
    """Test with good detail preservation"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = MemoryDetailPreservationGrader(model=model)

    result = await grader.aevaluate(
        observation="Cabinet 1 at (3.5, 2.1) contains 5 red apples.",
        memory="Cabinet 1 at coordinates (3.5, 2.1): 5 red apples.",
        task_context="Task: Precise inventory",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good detail preservation


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_detail_preservation_with_history():
    """Test detail preservation with history"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = MemoryDetailPreservationGrader(model=model)

    history = [
        {"observation": "Room A has 10 items", "memory": "Room A: 10 items"},
    ]

    result = await grader.aevaluate(
        observation="Drawer 5 in Room B at position (10.2, 5.8) contains key-A123.",
        memory="Found a key somewhere.",  # Lost critical details
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
