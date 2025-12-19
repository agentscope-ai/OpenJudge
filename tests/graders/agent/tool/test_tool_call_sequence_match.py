# -*- coding: utf-8 -*-
"""
Test Tool Call Sequence Match Grader

Tests for the ToolCallSequenceMatchGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.agent.tool.tool_call_sequence_match import (
    ToolCallSequenceMatchGrader,
)


def test_tool_call_sequence_match_grader_creation():
    """Test creating a ToolCallSequenceMatchGrader instance"""
    grader = ToolCallSequenceMatchGrader(strict_mode=True)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_call_sequence"
    assert grader.strict_mode is True


def test_tool_call_sequence_match_grader_loose_mode():
    """Test creating grader in loose mode"""
    grader = ToolCallSequenceMatchGrader(strict_mode=False)

    assert grader.strict_mode is False


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_empty_messages():
    """Test with empty messages"""
    grader = ToolCallSequenceMatchGrader(strict_mode=True)

    result = await grader.aevaluate(messages=[], reference_tool_calls=[])

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 1.0  # Empty sequences match


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_exact_match():
    """Test with exact matching sequence"""
    grader = ToolCallSequenceMatchGrader(strict_mode=True)

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
    ]

    reference_tool_calls = [
        [{"name": "search", "arguments": {"query": "test"}}],
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score > 0.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_mismatch():
    """Test with mismatched sequence"""
    grader = ToolCallSequenceMatchGrader(strict_mode=True)

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
    ]

    reference_tool_calls = [
        [{"name": "calculate", "parameters": {"value": 42}}],
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    # Score should reflect mismatch
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_loose_mode_matching():
    """Test loose mode (only tool names)"""
    grader = ToolCallSequenceMatchGrader(strict_mode=False)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "different"}',
                    },
                },
            ],
        },
    ]

    reference_tool_calls = [
        [{"name": "search", "arguments": {"query": "test"}}],
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    # In loose mode, should match on tool name only
    assert result.score > 0.0


def test_tool_call_sequence_match_grader_extract_predicted_tool_sequence():
    """Test extracting predicted tool sequence from messages"""
    grader = ToolCallSequenceMatchGrader()

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
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "analyze",
                        "arguments": '{"data": "result"}',
                    },
                },
            ],
        },
    ]

    sequence = grader.extract_predicted_tool_sequence(messages)

    assert len(sequence) == 2
    assert 0 in sequence
    assert 1 in sequence
    assert sequence[0][0]["name"] == "search"
    assert sequence[1][0]["name"] == "analyze"
