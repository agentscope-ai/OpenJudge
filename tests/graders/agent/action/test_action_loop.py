# -*- coding: utf-8 -*-
"""
Test Action Loop Detection Grader

Tests for the ActionLoopDetectionGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.agent.action.action_loop import ActionLoopDetectionGrader


def test_action_loop_detection_grader_creation():
    """Test creating an ActionLoopDetectionGrader instance"""
    grader = ActionLoopDetectionGrader()

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "action_loop_detection"


@pytest.mark.asyncio
async def test_action_loop_detection_grader_no_actions():
    """Test with no actions in messages"""
    grader = ActionLoopDetectionGrader()

    result = await grader.aevaluate(messages=[], similarity_threshold=1.0)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 1.0  # No actions means no loops


@pytest.mark.asyncio
async def test_action_loop_detection_grader_with_loops():
    """Test detecting repetitive actions"""
    grader = ActionLoopDetectionGrader()

    # Messages with repetitive tool calls
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
        {"role": "function", "content": "result 1"},
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
        {"role": "function", "content": "result 2"},
    ]

    result = await grader.aevaluate(messages=messages, similarity_threshold=1.0)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score < 1.0  # Should detect similarity
    assert "similar" in result.reason.lower()


@pytest.mark.asyncio
async def test_action_loop_detection_grader_no_loops():
    """Test with different actions (no loops)"""
    grader = ActionLoopDetectionGrader()

    # Messages with different tool calls
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
        {"role": "function", "content": "result 1"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculate",
                        "arguments": '{"value": 42}',
                    }
                }
            ],
        },
        {"role": "function", "content": "result 2"},
    ]

    result = await grader.aevaluate(messages=messages, similarity_threshold=1.0)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score >= 0.0

