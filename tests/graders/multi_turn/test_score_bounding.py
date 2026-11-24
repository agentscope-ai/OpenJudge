#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-cutting tests: multi-turn graders bound model scores to the documented 1-5 scale.

All seven multi-turn graders document a 1-5 score and prompt the model for one,
but the model can return anything. The returned ``score`` must stay on the
documented scale while ``metadata["raw_score"]`` preserves the model's value
for downstream audits (the pattern established for TrajectoryAccuracyGrader
and ToolCallAccuracyGrader).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openjudge.graders.multi_turn.anaphora_resolution_grader import (
    AnaphoraResolutionGrader,
)
from openjudge.graders.multi_turn.context_memory_grader import ContextMemoryGrader
from openjudge.graders.multi_turn.instruction_clarification_grader import (
    InstructionClarificationGrader,
)
from openjudge.graders.multi_turn.proactive_interaction_grader import (
    ProactiveInteractionGrader,
)
from openjudge.graders.multi_turn.response_repetition_grader import (
    ResponseRepetitionGrader,
)
from openjudge.graders.multi_turn.self_correction_grader import SelfCorrectionGrader
from openjudge.graders.multi_turn.topic_switch_grader import TopicSwitchGrader
from openjudge.graders.schema import GraderError

GRADER_CLASSES = [
    AnaphoraResolutionGrader,
    ContextMemoryGrader,
    InstructionClarificationGrader,
    ProactiveInteractionGrader,
    ResponseRepetitionGrader,
    SelfCorrectionGrader,
    TopicSwitchGrader,
]

HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


@pytest.mark.unit
@pytest.mark.parametrize("grader_class", GRADER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "model_score,expected",
    [(0.3, 1.0), (1.0, 1.0), (3.5, 3.5), (5.0, 5.0), (42, 5.0), (-2.0, 1.0)],
)
async def test_score_bounded_to_documented_scale(grader_class, model_score, expected):
    """score is bounded to 1-5; metadata["raw_score"] keeps the model's value"""
    grader = grader_class(model=AsyncMock())
    grader.model.achat = AsyncMock(
        return_value=SimpleNamespace(parsed={"score": model_score, "reason": "Synthetic offline response"}),
    )

    result = await grader.aevaluate(response="some response", history=HISTORY)

    assert result.score == expected
    assert result.metadata["raw_score"] == model_score


@pytest.mark.unit
@pytest.mark.parametrize("grader_class", GRADER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("model_score", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_model_score_returns_error(grader_class, model_score):
    """NaN and the infinities cannot be bounded, so they route through the
    error path instead of becoming a perfect 5.0"""
    grader = grader_class(model=AsyncMock())
    grader.model.achat = AsyncMock(
        return_value=SimpleNamespace(parsed={"score": model_score, "reason": "Synthetic offline response"}),
    )

    result = await grader.aevaluate(response="some response", history=HISTORY)

    assert isinstance(result, GraderError)
    assert "non-finite score" in result.error
