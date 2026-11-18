# -*- coding: utf-8 -*-
"""
Test Action Loop Detection Grader

Tests for the ActionLoopDetectionGrader class functionality.
"""

import pytest

from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader


def test_action_loop_detection_grader_creation():
    """Test creating an ActionLoopDetectionGrader instance"""
    grader = ActionLoopDetectionGrader()

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "action_loop_detection"


@pytest.mark.asyncio
async def test_action_loop_detection_grader_no_actions():
    """Test with no actions in messages"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    result = await grader.aevaluate(messages=[])

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 1.0  # No actions means no loops


@pytest.mark.asyncio
async def test_action_loop_detection_grader_with_loops():
    """Test detecting repetitive actions"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    # Messages with repetitive tool calls
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                },
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
                    },
                },
            ],
        },
        {"role": "function", "content": "result 2"},
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score < 1.0  # Should detect similarity
    assert "similar" in result.reason.lower()


@pytest.mark.asyncio
async def test_action_loop_detection_grader_no_loops():
    """Test with different actions (no loops)"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    # Messages with different tool calls
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test1"}',
                    },
                },
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
                    },
                },
            ],
        },
        {"role": "function", "content": "result 2"},
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    assert result.score >= 0.0


def _tool_call_messages(arguments_list):
    messages = []
    for index, arguments in enumerate(arguments_list):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "function": {"name": "search", "arguments": arguments},
                    },
                ],
            },
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{index}", "content": "result"})
    return messages


@pytest.mark.asyncio
async def test_action_loop_detection_grader_accepts_dict_arguments():
    """Already-parsed argument dicts are compared like JSON strings"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    repeated = await grader.aevaluate(
        messages=_tool_call_messages([{"query": "test", "top_k": 3}, {"top_k": 3, "query": "test"}]),
    )
    distinct = await grader.aevaluate(
        messages=_tool_call_messages([{"query": "test"}, {"query": "other"}]),
    )

    assert repeated.score == 0.0
    assert repeated.metadata["signatures"] == ["search: query: test, top_k: 3"] * 2
    assert distinct.score == 1.0


@pytest.mark.asyncio
async def test_action_loop_detection_grader_ignores_nested_key_order():
    """Nested argument objects with the same content are the same action"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    result = await grader.aevaluate(
        messages=_tool_call_messages(
            [
                '{"filter": {"lang": "en", "year": 2024}, "query": "test"}',
                '{"query": "test", "filter": {"year": 2024, "lang": "en"}}',
            ],
        ),
    )

    assert result.score == 0.0
    assert result.metadata["similar_pair_count"] == 1


@pytest.mark.asyncio
async def test_action_loop_detection_grader_keeps_unparsable_arguments():
    """Arguments that are not a JSON object still contribute their raw text"""
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    result = await grader.aevaluate(messages=_tool_call_messages(["not json", "[1, 2]", ""]))

    assert result.metadata["signatures"] == ["search: not json", "search: [1, 2]", "search"]
    assert result.score == 1.0
