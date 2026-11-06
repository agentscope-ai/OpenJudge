# -*- coding: utf-8 -*-
"""Tests for the reasoning tool-call format grader."""

import pytest

from openjudge.graders.format.reasoning_tool_format import ReasoningToolCallFormatGrader


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    ["null", "true", "42", '"name arguments"', '["name", "arguments"]', "{}"],
)
async def test_invalid_tool_call_payload_scores_zero(payload: str) -> None:
    """Report invalid model output using the regular score and reason."""
    grader = ReasoningToolCallFormatGrader()

    result = await grader.aevaluate(response=f"<think>Need a tool</think><tool_call>{payload}</tool_call>")

    assert result.score == 0.0
    assert result.metadata["valid_tool_call_json"] is False
    assert result.reason == "Invalid JSON format in <tool_call> tags"


@pytest.mark.asyncio
async def test_valid_tool_call_payload_scores_one() -> None:
    """Keep accepting a tool-call object with both required fields."""
    grader = ReasoningToolCallFormatGrader()

    result = await grader.aevaluate(
        response='<think>Need a tool</think><tool_call>{"name": "lookup", "arguments": {}}</tool_call>',
    )

    assert result.score == 1.0
    assert result.metadata["valid_tool_call_json"] is True
