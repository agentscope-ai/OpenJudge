# -*- coding: utf-8 -*-
"""Preserve unmodified model scores for downstream grader audits (offline)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.agent.trajectory.trajectory_accuracy import (
    TrajectoryAccuracyGrader,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw_score,expected",
    [(-99.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.5, 2.5), (3.0, 3.0), (99.0, 3.0)],
)
def test_raw_score_retains_model_value(raw_score: float, expected: float) -> None:
    """Keep compatibility for score while exposing the pre-clamp value in metadata."""
    parsed = GraderScore(name="trajectory_accuracy", score=raw_score, reason="Synthetic offline response")
    grader = TrajectoryAccuracyGrader(model=AsyncMock())
    with patch("openjudge.graders.llm_grader.LLMGrader._aevaluate", new=AsyncMock(return_value=parsed)):
        result = asyncio.run(grader.aevaluate(messages=[{"role": "user", "content": "Synthetic task"}]))
    assert result.score == expected
    assert result.metadata["raw_score"] == raw_score
    assert result.reason == "Synthetic offline response"
