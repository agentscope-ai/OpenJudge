# -*- coding: utf-8 -*-
"""
Test Memory Retrieval Effectiveness Grader

Tests for the MemoryRetrievalEffectivenessGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import MemoryRetrievalEffectivenessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_memory_retrieval_effectiveness_grader_creation():
    """Test creating a MemoryRetrievalEffectivenessGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "memory_retrieval_effectiveness"


def test_memory_retrieval_effectiveness_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = MemoryRetrievalEffectivenessGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_effectiveness_poor():
    """Test detecting poor memory retrieval effectiveness"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    # Test case where plan ignores known information from memory (poor effectiveness)
    result = await grader.aevaluate(
        plan="I will search for the key in drawer 1.",
        observation="You are in the room.",
        memory="The key was already found in drawer 1 in step 3. Key is in inventory.",
        task_context="Task: Use the key to unlock the door",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor effectiveness


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_effectiveness_good():
    """Test with good memory retrieval effectiveness"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    result = await grader.aevaluate(
        plan="I will use the key I found earlier to unlock the door.",
        observation="You are near the locked door. Key is in inventory.",
        memory="Key-A was found in drawer 1.",
        task_context="Task: Unlock the door",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good effectiveness


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_effectiveness_with_history():
    """Test memory retrieval effectiveness with history"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    history = [
        {"observation": "Found key in drawer 1", "memory": "Key in drawer 1"},
        {"observation": "Picked up key", "memory": "Key in inventory"},
    ]

    result = await grader.aevaluate(
        plan="I should search all drawers to find a key.",  # Ignoring memory - poor effectiveness
        observation="Standing in the room with key in inventory.",
        memory="Key is in inventory.",
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
