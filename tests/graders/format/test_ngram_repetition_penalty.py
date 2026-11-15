# -*- coding: utf-8 -*-
"""Public scoring regressions for repetition penalty thresholds."""

import pytest

from openjudge.graders.format.ngram_repetition_penalty import (
    NgramRepetitionPenaltyGrader,
)
from openjudge.utils.tokenizer import TokenizerEnum


@pytest.mark.unit
@pytest.mark.parametrize(
    "n,response,min_scaling,expected_rate,expected_score",
    [
        (1, "alpha beta gamma alpha", 0.5, 0.25, 0.0),
        (1, "alpha beta alpha beta", 0.5, 0.5, 0.0),
        (2, "alpha alpha alpha alpha beta", 0.5, 0.5, 0.0),
        (3, "alpha alpha alpha alpha alpha beta", 0.5, 0.5, 0.0),
        (1, "alpha beta alpha beta alpha beta alpha beta", 0.75, 0.75, 0.0),
        (1, "alpha beta alpha beta alpha", 0.5, 0.6, -0.4),
        (1, "alpha beta alpha beta alpha beta alpha beta", 0.5, 0.75, -1.0),
        (1, "alpha beta gamma delta", 0.0, 0.0, 0.0),
        (1, "alpha beta alpha beta", 0.0, 0.5, -1.0),
    ],
)
async def test_soft_penalty_starts_above_min_scaling(
    n: int, response: str, min_scaling: float, expected_rate: float, expected_score: float
) -> None:
    """Score actual tokenized responses without penalizing the threshold itself."""
    grader = NgramRepetitionPenaltyGrader(
        n=n,
        use_soft_penalty=True,
        min_scaling=min_scaling,
        max_penalty=-2.0,
        tokenizer_type=TokenizerEnum.simple,
    )

    result = await grader.aevaluate(response=response)

    assert result.score == pytest.approx(expected_score)
    assert result.metadata["penalty"] == pytest.approx(expected_score)
    assert result.metadata["repetition_rate"] == pytest.approx(expected_rate)
    assert result.metadata["penalty_mode"] == "soft"


@pytest.mark.unit
@pytest.mark.parametrize(
    "response,expected_score",
    [("alpha beta alpha beta", 0.0), ("alpha beta alpha beta alpha beta alpha beta", -0.25)],
)
async def test_hard_penalty_threshold_is_unchanged(response: str, expected_score: float) -> None:
    """Hard penalties still apply only above their configured threshold."""
    grader = NgramRepetitionPenaltyGrader(n=1, penalty_threshold=0.5, tokenizer_type=TokenizerEnum.simple)

    result = await grader.aevaluate(response=response)

    assert result.score == pytest.approx(expected_score)
    assert result.metadata["penalty_mode"] == "hard"
