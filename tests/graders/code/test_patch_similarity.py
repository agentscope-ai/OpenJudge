#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for PatchSimilarityGrader.

Tests patch similarity calculation using difflib.SequenceMatcher.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_patch_similarity.py -v
    ```
"""

import pytest

from openjudge.graders.code.patch_similarity import PatchSimilarityGrader


@pytest.mark.unit
class TestPatchSimilarityGraderUnit:
    """Unit tests for PatchSimilarityGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        grader = PatchSimilarityGrader()
        assert grader.name == "patch_similarity"

    @pytest.mark.asyncio
    async def test_identical_patches(self):
        """Test identical patches return score of 1.0"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="def add(a, b):\n    return a + b",
            reference_response="def add(a, b):\n    return a + b",
        )

        assert result.score == 1.0
        assert result.metadata["similarity"] == 1.0

    @pytest.mark.asyncio
    async def test_completely_different_patches(self):
        """Test completely different patches return low score"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="def foo():\n    pass",
            reference_response="class Bar:\n    def baz(self):\n        return 42",
        )

        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_slightly_different_patches(self):
        """Test patches with minor differences"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="def add(a, b):\n    return a + b",
            reference_response="def add(x, y):\n    return x + y",
        )

        # Should have high similarity (only variable names differ)
        assert 0.5 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_empty_patches(self):
        """Test both empty patches"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="",
            reference_response="",
        )

        # Two empty strings are identical
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_empty_patch(self):
        """Test one empty and one non-empty patch"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="",
            reference_response="def add(a, b):\n    return a + b",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="hello",
            reference_response="world",
        )

        assert "similarity" in result.metadata
        assert "response" in result.metadata
        assert "reference_response" in result.metadata
        assert "opcodes" in result.metadata
        assert result.metadata["response"] == "hello"
        assert result.metadata["reference_response"] == "world"

    @pytest.mark.asyncio
    async def test_symmetry_of_similarity(self):
        """Test that similarity is symmetric (swapping response and reference gives same score)"""
        grader = PatchSimilarityGrader()
        result1 = await grader.aevaluate(
            response="def foo():\n    pass",
            reference_response="def bar():\n    pass",
        )
        result2 = await grader.aevaluate(
            response="def bar():\n    pass",
            reference_response="def foo():\n    pass",
        )

        # SequenceMatcher.ratio() is symmetric
        assert result1.score == result2.score

    @pytest.mark.asyncio
    async def test_single_char_difference(self):
        """Test patches that differ by a single character"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="return a + b",
            reference_response="return a - b",
        )

        # High similarity, only one char differs
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_multiline_patch_difference(self):
        """Test patches with multiple line differences"""
        grader = PatchSimilarityGrader()
        response_patch = """def process(data):
    result = []
    for item in data:
        result.append(transform(item))
    return result"""

        reference_patch = """def process(data):
    return [transform(item) for item in data]"""

        result = await grader.aevaluate(
            response=response_patch,
            reference_response=reference_patch,
        )

        # Should have moderate similarity (same function name, different implementation)
        assert 0.3 < result.score < 0.8

    @pytest.mark.asyncio
    async def test_reason_message_format(self):
        """Test that reason message contains similarity score"""
        grader = PatchSimilarityGrader()
        result = await grader.aevaluate(
            response="abc",
            reference_response="abc",
        )

        assert "Patch similarity" in result.reason
        assert "1.000" in result.reason
