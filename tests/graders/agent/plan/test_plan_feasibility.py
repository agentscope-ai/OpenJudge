# -*- coding: utf-8 -*-
"""
Test Plan Feasibility Grader

Tests for the PlanFeasibilityGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.predefined.agent import PlanFeasibilityGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_plan_feasibility_grader_creation():
    """Test creating a PlanFeasibilityGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = PlanFeasibilityGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "plan_feasibility"


def test_plan_feasibility_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = PlanFeasibilityGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_feasibility_poor():
    """Test detecting poor plan feasibility (impossible action)"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = PlanFeasibilityGrader(model=model)

    # Test case with infeasible plan (using object before obtaining it)
    result = await grader.aevaluate(
        plan="I will use the key to unlock the door.",
        observation="The drawer is closed. You don't have any items.",
        memory="The key is inside the drawer, but the drawer is not opened yet.",
        task_context="Task: Unlock the door to exit",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect poor feasibility


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_feasibility_good():
    """Test with good plan feasibility"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = PlanFeasibilityGrader(model=model)

    result = await grader.aevaluate(
        plan="I will first open the drawer to get the key, then unlock the door.",
        observation="Drawer is closed. Key is inside.",
        memory="Key is in drawer 1.",
        task_context="Task: Unlock the door",
    )

    assert result is not None
    assert result.score == 1.0  # Should have good feasibility


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_feasibility_with_history():
    """Test plan feasibility with history"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = PlanFeasibilityGrader(model=model)

    history = [
        {"observation": "Door is locked", "plan": "Need to find key"},
        {"observation": "Drawer is closed", "plan": "Will check drawer"},
    ]

    result = await grader.aevaluate(
        plan="I will close the door.",  # Door is already locked, can't close - infeasible
        observation="Door is locked, cannot be closed further.",
        memory="Door is locked.",
        history_steps=history,
    )

    assert result is not None
    assert result.score == 0.0
