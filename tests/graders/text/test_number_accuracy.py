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

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("10-20 units", [10.0, 20.0]),
            ("2024-05-01", [2024.0, 5.0, 1.0]),
            ("a 3-1 win", [3.0, 1.0]),
            ("1.5-2.5", [1.5, 2.5]),
            ("1e-5-2e-5", [1e-5, 2e-5]),
        ],
    )
    async def test_hyphen_after_a_digit_is_not_a_minus_sign(self, text: str, expected: list) -> None:
        """A hyphen joining two numbers (range, date, score) does not negate the second one."""
        assert NumberAccuracyGrader()._extract_numbers(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("-7", [-7.0]),
            ("fell to -5 degrees", [-5.0]),
            ("x=-3 and y=4", [-3.0, 4.0]),
            ("(-2.5)", [-2.5]),
            ("between -10 and -20", [-10.0, -20.0]),
            ("温度为-5度", [-5.0]),
        ],
    )
    async def test_minus_sign_is_kept_when_not_after_a_digit(self, text: str, expected: list) -> None:
        """A hyphen that does not follow a digit still marks a negative number."""
        assert NumberAccuracyGrader()._extract_numbers(text) == expected

    @pytest.mark.parametrize(
        "response,reference",
        [
            ("The answer is 10 to 20 units.", "The answer is 10-20 units."),
            ("Released on 2024/05/01.", "Released on 2024-05-01."),
            ("Pages 10-20 cover it.", "Pages 10–20 cover it."),
        ],
    )
    async def test_ranges_and_dates_written_differently_score_as_correct(self, response: str, reference: str) -> None:
        """The same numbers joined by a hyphen, a word, a slash or an en dash all match."""
        result = await NumberAccuracyGrader().aevaluate(response=response, reference_response=reference)

        assert result.score == 1.0
        assert result.metadata["response_numbers"] == result.metadata["reference_response_numbers"]

    async def test_dropped_minus_sign_is_still_wrong(self) -> None:
        """A response that loses a real minus sign is not rescued by the range rule."""
        result = await NumberAccuracyGrader().aevaluate(
            response="It fell to 5 degrees.",
            reference_response="It fell to -5 degrees.",
        )

        assert result.score == 0.0

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
