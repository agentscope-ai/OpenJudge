# -*- coding: utf-8 -*-
"""
Unit Tests for JSON Matching and Validation Graders

Tests for JsonMatchGrader and JsonValidatorGrader.
"""

import json

import pytest

from openjudge.graders.format.json.json_match import JsonMatchGrader


class TestJsonMatchGrader:
    """Test JsonMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match_simple(self):
        """Test exact match for simple JSON objects"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": "Alice", "age": 30}',
            response='{"name": "Alice", "age": 30}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_exact_match_different_order(self):
        """Test that dict key order doesn't matter"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": "Alice", "age": 30}',
            response='{"age": 30, "name": "Alice"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_no_match_different_values(self):
        """Test no match when values differ"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": "Alice", "age": 30}',
            response='{"name": "Bob", "age": 30}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_no_match_missing_key(self):
        """Test no match when key is missing"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": "Alice", "age": 30}',
            response='{"name": "Alice"}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_ignore_extra_keys(self):
        """Test ignore_extra_keys option"""
        grader = JsonMatchGrader(ignore_extra_keys=True)

        result = await grader.aevaluate(
            reference_response='{"name": "Alice"}',
            response='{"name": "Alice", "age": 30, "city": "NYC"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_match_same_order(self):
        """Test list matching with same order"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(reference_response="[1, 2, 3]", response="[1, 2, 3]")

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_no_match_different_order(self):
        """Test list doesn't match with different order (strict_order=True)"""
        grader = JsonMatchGrader(strict_order=True)

        result = await grader.aevaluate(reference_response="[1, 2, 3]", response="[3, 2, 1]")

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_list_match_different_order_allowed(self):
        """Test list matches with different order when strict_order=False"""
        grader = JsonMatchGrader(strict_order=False)

        result = await grader.aevaluate(reference_response="[1, 2, 3]", response="[3, 2, 1]")

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_no_match_different_length(self):
        """Test lists with different lengths don't match"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(reference_response="[1, 2, 3]", response="[1, 2]")

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_nested_structure_match(self):
        """Test nested JSON structure matching"""
        grader = JsonMatchGrader()

        reference = json.dumps(
            {
                "user": {
                    "name": "Alice",
                    "contacts": {"email": "alice@example.com", "phone": "123-4567"},
                },
                "active": True,
            },
        )

        response = json.dumps(
            {
                "user": {
                    "name": "Alice",
                    "contacts": {"email": "alice@example.com", "phone": "123-4567"},
                },
                "active": True,
            },
        )

        result = await grader.aevaluate(reference_response=reference, response=response)

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_nested_list_in_dict(self):
        """Test nested lists in dict"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"items": [1, 2, 3], "name": "test"}',
            response='{"items": [1, 2, 3], "name": "test"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_invalid_response_json(self):
        """Test handling of invalid response JSON"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": "Alice"}',
            response="not valid json",
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False
        assert result.metadata["error"] == "response_parse_error"

    @pytest.mark.asyncio
    async def test_invalid_reference_json(self):
        """Test handling of invalid reference JSON"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response="not valid json",
            response='{"name": "Alice"}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False
        assert result.metadata["error"] == "reference_parse_error"

    @pytest.mark.asyncio
    async def test_null_values(self):
        """Test handling of null values"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"name": null}',
            response='{"name": null}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_boolean_values(self):
        """Test boolean value matching"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"active": true, "deleted": false}',
            response='{"active": true, "deleted": false}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_boolean_vs_number_not_matched(self):
        """Test that booleans are not matched against numbers.

        JSON treats booleans as a distinct type from numbers, but Python's
        ``==`` considers True == 1 and False == 0. The grader must not let
        {"a": true} match {"a": 1}.
        """
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"active": true, "deleted": false}',
            response='{"active": 1, "deleted": 0}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_number_vs_boolean_not_matched(self):
        """Test the mirrored case: numbers must not match booleans"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"count": 1}',
            response='{"count": true}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_boolean_vs_number_in_list_not_matched(self):
        """Test bool/number confusion inside lists with strict order"""
        grader = JsonMatchGrader(strict_order=True)

        result = await grader.aevaluate(
            reference_response="[true, 1]",
            response="[1, true]",
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_int_vs_float_numeric_equivalence(self):
        """Test that numerically equal ints and floats still match"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"value": 1}',
            response='{"value": 1.0}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_number_types(self):
        """Test different number types"""
        grader = JsonMatchGrader()

        result = await grader.aevaluate(
            reference_response='{"int": 42, "float": 3.14}',
            response='{"int": 42, "float": 3.14}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_empty_structures(self):
        """Test empty dict and list"""
        grader = JsonMatchGrader()

        # Empty dict
        result = await grader.aevaluate(reference_response="{}", response="{}")
        assert result.score == 1.0

        # Empty list
        result = await grader.aevaluate(reference_response="[]", response="[]")
        assert result.score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
