# -*- coding: utf-8 -*-
"""
Unit Tests for Per-Call Parameter Precedence in SimilarityGrader

The grader documents three layers of parameters, in increasing priority:
algorithm defaults, constructor configuration, then per-call keyword
arguments. These tests pin that order: an argument passed to
``aevaluate`` must win over the same key stored on the instance.
"""

import pytest

from openjudge.graders.text.similarity import SimilarityGrader


class TestPerCallOverride:
    """Per-call kwargs must override constructor-level configuration."""

    @pytest.mark.asyncio
    async def test_per_call_override_beats_constructor_kwargs(self):
        """A per-call argument must win over an extra constructor kwarg.

        ``score_key`` is not a named constructor parameter, so it lands in
        ``self.kwargs``. Passing it per call is documented to override the
        constructor value.
        """
        grader = SimilarityGrader(algorithm="rouge1", score_key="precision")

        result = await grader.aevaluate(
            reference_response="alpha beta gamma",
            response="alpha beta gamma delta",
            score_key="recall",
        )

        assert result.metadata["score_key"] == "recall"
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_per_call_override_beats_constructor_kwargs_for_edit_distance(self):
        """The same precedence must hold for boolean algorithm options."""
        grader = SimilarityGrader(algorithm="edit_distance", normalize_by_length=False)

        result = await grader.aevaluate(
            reference_response="abc",
            response="abd",
            normalize_by_length=True,
        )

        assert result.metadata["normalize_by_length"] is True
        # Length-normalized similarity is 1 - 1/3; the un-normalized path
        # would return 1 / (1 + 1) = 0.5 instead.
        assert result.score == pytest.approx(2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_constructor_kwargs_still_apply_without_per_call_override(self):
        """Constructor configuration must keep working when nothing overrides it."""
        grader = SimilarityGrader(algorithm="rouge1", score_key="precision")

        result = await grader.aevaluate(
            reference_response="alpha beta gamma",
            response="alpha beta gamma delta",
        )

        assert result.metadata["score_key"] == "precision"
        assert result.score == pytest.approx(0.75)
