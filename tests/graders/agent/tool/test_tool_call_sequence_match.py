# -*- coding: utf-8 -*-
"""
Test Tool Call Sequence Match Grader

Tests for the ToolCallSequenceMatchGrader class functionality.
"""

import pytest

from openjudge.graders.agent.tool.tool_call_sequence_match import (
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


def test_tool_call_sequence_match_grader_metric_type_default():
    """Test that default metric_type is recall"""
    grader = ToolCallSequenceMatchGrader(strict_mode=False, use_jaccard_similarity=False)
    assert grader.metric_type == "recall"


def test_tool_call_sequence_match_grader_metric_type_precision():
    """Test creating grader with precision metric_type"""
    grader = ToolCallSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="precision",
    )
    assert grader.metric_type == "precision"


def test_tool_call_sequence_match_grader_invalid_metric_type():
    """Test that invalid metric_type raises ValueError"""
    with pytest.raises(ValueError, match="metric_type must be 'recall' or 'precision'"):
        ToolCallSequenceMatchGrader(metric_type="invalid")


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_recall_metric():
    """Test loose mode with recall metric (matched / reference)"""
    grader = ToolCallSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="recall",
    )

    # Predicted has 1 tool, reference has 2 tools, 1 match
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search", "arguments": "{}"}},
            ],
        },
    ]

    reference_tool_calls = [
        [
            {"name": "search", "arguments": {}},
            {"name": "calculate", "arguments": {}},
        ],
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    # Recall = 1 matched / 2 reference = 0.5
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_precision_metric():
    """Test loose mode with precision metric (matched / predicted)"""
    grader = ToolCallSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="precision",
    )

    # Predicted has 2 tools, reference has 1 tool, 1 match
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search", "arguments": "{}"}},
                {"function": {"name": "calculate", "arguments": "{}"}},
            ],
        },
    ]

    reference_tool_calls = [
        [
            {"name": "search", "arguments": {}},
        ],
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    # Precision = 1 matched / 2 predicted = 0.5
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_recall_vs_precision():
    """Test that recall and precision give different scores for same input"""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search", "arguments": "{}"}},
                {"function": {"name": "extra_tool", "arguments": "{}"}},
            ],
        },
    ]

    reference_tool_calls = [
        [
            {"name": "search", "arguments": {}},
        ],
    ]

    # Recall grader: 1 matched / 1 reference = 1.0
    recall_grader = ToolCallSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="recall",
    )
    recall_result = await recall_grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    # Precision grader: 1 matched / 2 predicted = 0.5
    precision_grader = ToolCallSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="precision",
    )
    precision_result = await precision_grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    assert recall_result.score == 1.0
    assert precision_result.score == 0.5
