#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the jaccard algorithm of SimilarityGrader."""

import pytest

from openjudge.graders.text.similarity import SimilarityGrader


@pytest.mark.asyncio
class TestJaccard:
    async def test_identical_inputs(self):
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="hello world",
            response="hello world",
        )
        assert result.score == 1.0

    async def test_partial_overlap(self):
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="hello world",
            response="hello there",
        )
        assert 0.0 < result.score < 1.0

    async def test_disjoint_inputs(self):
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="alpha beta",
            response="gamma delta",
        )
        assert result.score == 0.0

    async def test_empty_strings(self):
        """Both inputs empty strings: documented perfect-match semantics"""
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="",
            response="",
        )
        assert result.score == 1.0

    async def test_empty_reference(self):
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="",
            response="hello",
        )
        assert result.score == 0.0

    async def test_punctuation_only_both_sides_scores_zero(self):
        """Non-empty inputs whose tokens are all removed (punctuation only)
        must fail closed, not silently score a perfect match (see #199)."""
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="！！！。？",
            response="……——",
        )
        assert result.score == 0.0

    async def test_ngram_variant_punctuation_only_scores_zero(self):
        """The n-gram path shares the fail-closed behavior"""
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response="！！！。？",
            response="……——",
            use_ngrams=True,
            n=2,
        )
        assert result.score == 0.0

    @pytest.mark.parametrize(
        ("reference", "response", "expected"),
        [
            ("Paris", "Paris", 1.0),
            ("巴黎", "巴黎", 1.0),
            ("Paris", "London", 0.0),
            ("Paris", "Paris France", 0.0),
        ],
    )
    async def test_ngram_variant_texts_shorter_than_n(self, reference, response, expected):
        """Texts with fewer than n tokens are compared as one gram instead of scoring 0"""
        grader = SimilarityGrader(algorithm="jaccard")
        result = await grader.aevaluate(
            reference_response=reference,
            response=response,
            use_ngrams=True,
            n=2,
        )
        assert result.score == expected
