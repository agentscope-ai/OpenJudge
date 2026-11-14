# -*- coding: utf-8 -*-
"""
Tests for NumberAccuracyGrader
"""

from decimal import ROUND_FLOOR, Inexact, localcontext

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

    @pytest.mark.parametrize(
        "response,reference,tolerance,expected",
        [
            ("9007199254740992", "9007199254740993", 1e-6, 0.0),
            ("-9007199254740992", "-9007199254740993", 1e-6, 0.0),
            ("9007199254740993", "9.007199254740993e15", 0.0, 1.0),
            ("1e400", "10e399", 0.0, 1.0),
            ("1e400", "2e400", 1e-6, 0.0),
            ("1e-400", "0", 0.0, 0.0),
            ("1e-400", "2e-400", 0.0, 0.0),
            ("1e-400", "1e-400", 0.0, 1.0),
            ("0.000001", "0", 1e-6, 1.0),
            ("0.0000010000000000000000000000000001", "0", 1e-6, 0.0),
            ("0.0000009999999999999999999999999999", "0", 1e-6, 1.0),
            ("1.000000000000000000000000000001", "0", 1.0, 0.0),
            ("1", "-1e-400", 1.0, 0.0),
            ("1", "1e-400", 1.0, 1.0),
            ("1.0", "1", 0.0, 1.0),
            ("1.01", "1", 0.0, 0.0),
            ("1", "1", -1.0, 0.0),
            ("1", "2", float("inf"), 1.0),
            ("1", "1", float("-inf"), 0.0),
            ("1", "1", float("nan"), 0.0),
        ],
    )
    async def test_compare_original_numeric_values(
        self, response: str, reference: str, tolerance: float, expected: float
    ) -> None:
        """Compare tokens before float rounding, retaining float-valued metadata."""
        grader = NumberAccuracyGrader(tolerance=tolerance)

        result = await grader.aevaluate(response=response, reference_response=reference)

        assert result.score == expected
        assert result.metadata["correct_numbers"] == int(expected)
        assert result.metadata["response_numbers"] == [float(response)]
        assert result.metadata["reference_response_numbers"] == [float(reference)]

    async def test_comparison_is_independent_of_decimal_context(self) -> None:
        """A caller's decimal rounding and traps must not change the grade."""
        with localcontext() as context:
            context.prec = 2
            context.rounding = ROUND_FLOOR
            context.traps[Inexact] = True
            result = await NumberAccuracyGrader(tolerance=1).aevaluate(
                response="1.000000000000000000000000000001", reference_response="0"
            )

        assert result.score == 0.0

    async def test_precise_comparison_preserves_order_and_reference_count(self) -> None:
        """Partial scores still use ordered pairs and all reference numbers."""
        result = await NumberAccuracyGrader().aevaluate(
            response="9007199254740992 and 2.0",
            reference_response="9007199254740993 and 2 and 3",
        )

        assert result.score == 1 / 3
        assert result.metadata["correct_numbers"] == 1
        assert result.metadata["total_reference_response_numbers"] == 3
