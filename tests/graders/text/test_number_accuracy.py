# -*- coding: utf-8 -*-
"""
Tests for NumberAccuracyGrader
"""

import pytest

from openjudge.graders.text.number_accuracy import NumberAccuracyGrader


@pytest.mark.unit
class TestNumberAccuracyGrader:
    """Test cases for NumberAccuracyGrader"""

    async def test_extract_numbers_plain(self):
        """Plain integers and decimals are extracted unchanged."""
        grader = NumberAccuracyGrader()
        assert grader._extract_numbers("3.14159 and -2.718 and 42") == [3.14159, -2.718, 42.0]

    async def test_extract_scientific_notation(self):
        """Scientific notation is extracted as a single number, not split apart."""
        grader = NumberAccuracyGrader()
        assert grader._extract_numbers("1e-10") == [1e-10]
        assert grader._extract_numbers("2.5e-10") == [2.5e-10]
        assert grader._extract_numbers("6.02e23") == [6.02e23]

    async def test_extract_leading_decimal(self):
        """A leading decimal point must not be dropped."""
        grader = NumberAccuracyGrader()
        assert grader._extract_numbers("probability is .5") == [0.5]

    async def test_scientific_notation_matches_decimal_equivalent(self):
        """1e-10 and 0.0000000001 are the same number and should score 1.0."""
        grader = NumberAccuracyGrader(tolerance=1e-9)
        result = await grader.aevaluate(
            response="The result is 1e-10",
            reference_response="The result is 0.0000000001",
        )
        assert result.score == 1.0
        assert result.metadata["correct_numbers"] == 1

    async def test_leading_decimal_matches(self):
        """'.5' should be treated as 0.5 and match the reference."""
        grader = NumberAccuracyGrader(tolerance=1e-9)
        result = await grader.aevaluate(
            response="probability is .5",
            reference_response="probability is 0.5",
        )
        assert result.score == 1.0
