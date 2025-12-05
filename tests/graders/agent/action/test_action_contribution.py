# -*- coding: utf-8 -*-
"""
Test Step Contribution Grader

Tests for the StepContributionGrader class functionality.
"""

import pytest

from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum
from rm_gallery.core.graders.agent.action.action_contribution import ActionContributionGrader


def test_action_contribution_grader_creation():
    """Test creating a ActionContributionGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = ActionContributionGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "action_contribution"


def test_action_contribution_grader_creation_chinese():
    """Test creating a Chinese ActionContributionGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = ActionContributionGrader(model=model, language=LanguageEnum.ZH)

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_contribution_evaluation():
    """Test evaluating action contribution"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = ActionContributionGrader(model=model)

    result = await grader.aevaluate(
        user_query="What is the weather in Beijing?",
        tool_calls="search(query='Beijing weather')",
        tool_responses="Temperature: 25°C, Sunny",
        final_answer="The weather in Beijing is sunny with a temperature of 25°C.",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert 0.0 <= result.score <= 1.0


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_contribution_with_context():
    """Test action contribution with context"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = ActionContributionGrader(model=model)

    result = await grader.aevaluate(
        user_query="Find the best restaurant in Shanghai",
        context_before="User is looking for dining options",
        tool_calls="search_restaurants(city='Shanghai', rating='high')",
        tool_responses="Found 5 top-rated restaurants",
        context_after="Analyzing restaurant options",
        final_answer="I recommend Restaurant A with 5-star rating.",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert 0.0 <= result.score <= 1.0

