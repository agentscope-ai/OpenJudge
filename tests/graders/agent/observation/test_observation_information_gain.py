# -*- coding: utf-8 -*-
"""
Test Search Information Gain Grader

Tests for the SearchInformationGainGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.agent.observation.observation_information_gain import ObservationInformationGainGrader


def test_observation_information_gain_grader_creation():
    """Test creating a ObservationInformationGainGrader instance"""
    grader = ObservationInformationGainGrader(similarity_threshold=0.5)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "observation_information_gain"
    assert grader.similarity_threshold == 0.5


@pytest.mark.asyncio
async def test_observation_information_gain_no_observations():
    """Test with no actions in messages"""
    grader = ObservationInformationGainGrader()

    result = await grader.aevaluate(messages=[])

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Neutral score: unable to evaluate
    assert result.metadata.get("evaluable") is False


@pytest.mark.asyncio
async def test_observation_information_gain_diverse_observations():
    """Test with diverse observations (high information gain)"""
    grader = ObservationInformationGainGrader(similarity_threshold=0.5)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test1"}',
                    }
                }
            ],
        },
        {"role": "function", "content": "Information about topic A with unique details"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test2"}',
                    }
                }
            ],
        },
        {
            "role": "function",
            "content": "Completely different information about topic B with other details",
        },
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score > 0.0  # Should have positive information gain


@pytest.mark.asyncio
async def test_observation_information_gain_redundant_observations():
    """Test with redundant observations (low information gain)"""
    grader = ObservationInformationGainGrader(similarity_threshold=0.5)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    }
                }
            ],
        },
        {"role": "function", "content": "The same information repeated multiple times"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    }
                }
            ],
        },
        {"role": "function", "content": "The same information repeated multiple times"},
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    # Score should be lower due to redundancy
    assert 0.0 <= result.score <= 1.0

