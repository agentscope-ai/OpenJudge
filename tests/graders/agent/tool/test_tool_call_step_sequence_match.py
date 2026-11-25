# -*- coding: utf-8 -*-
"""
Test Tool Call Sequence Match Grader

Tests for the ToolCallStepSequenceMatchGrader class functionality.
"""

import json

import pytest

from openjudge.graders.agent.tool.tool_call_step_sequence_match import (
    ToolCallStepSequenceMatchGrader,
)


def test_tool_call_sequence_match_grader_creation():
    """Test creating a ToolCallStepSequenceMatchGrader instance"""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_call_sequence"
    assert grader.strict_mode is True


def test_tool_call_sequence_match_grader_loose_mode():
    """Test creating grader in loose mode"""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=False)

    assert grader.strict_mode is False


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_empty_messages():
    """Test with empty messages"""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True)

    result = await grader.aevaluate(messages=[], reference_tool_calls=[])

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 1.0  # Empty sequences match


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_exact_match():
    """Test with exact matching sequence"""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True)

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
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True)

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
    grader = ToolCallStepSequenceMatchGrader(strict_mode=False)

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
    grader = ToolCallStepSequenceMatchGrader()

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
    grader = ToolCallStepSequenceMatchGrader(strict_mode=False, use_jaccard_similarity=False)
    assert grader.metric_type == "recall"


def test_tool_call_sequence_match_grader_metric_type_precision():
    """Test creating grader with precision metric_type"""
    grader = ToolCallStepSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="precision",
    )
    assert grader.metric_type == "precision"


def test_tool_call_sequence_match_grader_invalid_metric_type():
    """Test that invalid metric_type raises ValueError"""
    with pytest.raises(ValueError, match="metric_type must be 'recall' or 'precision'"):
        ToolCallStepSequenceMatchGrader(metric_type="invalid")


@pytest.mark.asyncio
async def test_tool_call_sequence_match_grader_recall_metric():
    """Test loose mode with recall metric (matched / reference)"""
    grader = ToolCallStepSequenceMatchGrader(
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
    grader = ToolCallStepSequenceMatchGrader(
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
    recall_grader = ToolCallStepSequenceMatchGrader(
        strict_mode=False,
        use_jaccard_similarity=False,
        metric_type="recall",
    )
    recall_result = await recall_grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    # Precision grader: 1 matched / 2 predicted = 0.5
    precision_grader = ToolCallStepSequenceMatchGrader(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("predicted_queries", "reference_queries", "expected_score"),
    [
        (["A", "B"], ["A", "B"], 1.0),
        (["B", "A"], ["A", "B"], 1.0),
        (["A", "B", "C"], ["C", "A", "B"], 1.0),
        (["A", "B"], ["A", "A"], 0.5),
        (["A", "A"], ["A", "B"], 0.5),
        (["C", "B", "A"], ["A", "B"], 1.0),
        (["A"], ["A", "A"], 0.0),
        (["A", "A"], ["A", "A"], 1.0),
        (["A"], ["A"], 1.0),
    ],
)
async def test_strict_steps_match_repeated_tools_once(predicted_queries, reference_queries, expected_score):
    """Match same-name calls by parameters without reusing predictions."""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True, use_jaccard_similarity=False)
    predicted = [{"name": "search", "arguments": {"query": query}} for query in predicted_queries]
    reference = [{"name": "search", "arguments": {"query": query}} for query in reference_queries]

    result = await grader.aevaluate(
        messages=[{"role": "assistant", "tool_calls": predicted}],
        reference_tool_calls=[reference],
    )

    assert result.score == pytest.approx(expected_score)


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_predictions", [False, True])
async def test_strict_steps_maximize_total_parameter_similarity(reverse_predictions):
    """An early tied match must not prevent a better total assignment."""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True, use_jaccard_similarity=False)
    # The first reference ties on either prediction (2/3). The second needs
    # the first prediction to score 1 rather than 0.
    reference = [
        {"name": "search", "arguments": {"a": "A", "b": "B"}},
        {"name": "search", "arguments": {"a": "A"}},
    ]
    predicted = [
        {"name": "search", "arguments": {"a": "A"}},
        {"name": "search", "arguments": {"b": "B"}},
    ]
    if reverse_predictions:
        predicted.reverse()

    result = await grader.aevaluate(
        messages=[{"role": "assistant", "tool_calls": predicted}],
        reference_tool_calls=[reference],
    )

    assert result.score == pytest.approx(5 / 6)


@pytest.mark.asyncio
async def test_strict_steps_keep_different_tool_names_separate():
    """Parameter matches cannot cross tool names or assistant turns."""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True, use_jaccard_similarity=False)
    result = await grader.aevaluate(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "search", "arguments": {"q": "B"}},
                    {"name": "write", "arguments": {"q": "A"}},
                ],
            },
            {"role": "assistant", "tool_calls": [{"name": "search", "arguments": {"q": "A"}}]},
        ],
        reference_tool_calls=[
            [{"name": "search", "arguments": {"q": "A"}}, {"name": "write", "arguments": {"q": "B"}}],
            [{"name": "search", "arguments": {"q": "B"}}],
        ],
    )

    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "predicted, reference, expected",
    [
        ({"city": "杭州", "units": "metric"}, {"units": "metric", "city": "杭州"}, 1.0),
        (
            {"filter": {"limit": 3, "tags": ["a", "b"]}},
            {"filter": {"tags": ["a", "b"], "limit": 3}},
            1.0,
        ),
        ({"items": [{"a": 1, "b": 2}]}, {"items": [{"b": 2, "a": 1}]}, 1.0),
        ({"city": "杭州"}, {"city": "上海"}, 0.0),
        ({"tags": ["a", "b"]}, {"tags": ["b", "a"]}, 0.0),
    ],
)
async def test_strict_jaccard_ignores_only_object_key_order(as_json, predicted, reference, expected):
    """Equivalent argument objects match, while changed values and array order do not."""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True, use_jaccard_similarity=True)
    result = await grader.aevaluate(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": json.dumps(predicted) if as_json else predicted}},
                ],
            },
        ],
        reference_tool_calls=[[{"name": "lookup", "arguments": reference}]],
    )

    assert result.score == expected
    assert result.metadata["predicted_tool_count"] == 1
    assert result.metadata["reference_tool_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "predicted, reference, expected",
    [
        ({"origin": "上海", "destination": "北京"}, {"origin": "北京", "destination": "上海"}, 0.0),
        ({"a": 2, "b": 1}, {"a": 1, "b": 2}, 0.0),
        ({"origin": "destination", "北京": "上海"}, {"origin": "北京", "destination": "上海"}, 0.0),
        ({"a": 1, "b": 3}, {"a": 1, "b": 2}, 0.5),
        ({"filter": {"year": 2024, "lang": "en"}}, {"filter": {"lang": "en", "year": 2024}}, 1.0),
        ({"tags": ["b", "a"]}, {"tags": ["a", "b"]}, 0.0),
        ({"filter": "[true]"}, {"filter": [True]}, 0.0),
        ({"filter": '{"year": 2024}'}, {"filter": {"year": 2024}}, 0.0),
        ({"limit": "1"}, {"limit": 1}, 0.0),
        ({"filter": [True]}, {"filter": [True]}, 1.0),
    ],
)
async def test_strict_steps_compare_parameters_as_key_value_pairs(predicted, reference, expected):
    """Values only match under their own key, and nested object key order is ignored."""
    grader = ToolCallStepSequenceMatchGrader(strict_mode=True, use_jaccard_similarity=False)
    result = await grader.aevaluate(
        messages=[{"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": predicted}]}],
        reference_tool_calls=[[{"name": "lookup", "arguments": reference}]],
    )

    assert result.score == pytest.approx(expected)
