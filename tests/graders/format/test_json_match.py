# -*- coding: utf-8 -*-
"""
Unit Tests for JSON Matching and Validation Graders

Tests for JsonMatchGrader and JsonValidatorGrader.
"""

import json
from itertools import permutations
from typing import Any

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


class TestUnorderedJsonMatching:
    """Unordered matching must assign distinct compatible reference records."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reference_order", list(permutations(range(3))))
    @pytest.mark.parametrize("response_order", list(permutations(range(3))))
    async def test_overlapping_records_are_order_independent(
        self, reference_order: tuple[int, ...], response_order: tuple[int, ...]
    ) -> None:
        """Every permutation of the same multiset must receive full credit."""
        records = [{}, {"a": 1}, {"a": 1, "b": 2}]
        reference = json.dumps([records[i] for i in reference_order])
        response = json.dumps([records[i] for i in response_order])
        result = await JsonMatchGrader(strict_order=False, ignore_extra_keys=True).aevaluate(
            reference_response=reference, response=response
        )
        assert result.score == 1.0
        assert result.metadata == {"matched": True, "strict_order": False, "ignore_extra_keys": True}
        assert records == [{}, {"a": 1}, {"a": 1, "b": 2}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reference,response,expected",
        [
            ([{"a": 1}, {"a": 1, "b": 2}], [{"a": 1, "b": 2}, {"a": 1}], 1.0),
            ([{"a": 1}, {"a": 1}], [{"a": 1}, {"a": 2}], 0.0),
            ([{}, {"a": 1}, {"a": 1}], [{"a": 1}, {"a": 2}, {"a": 2}], 0.0),
            ([{"a": 1}, {"a": 1}], [{"a": 1}, {"a": 1}], 1.0),
            ([{"a": 1}], [{"a": 1}, {"a": 1}], 0.0),
            ({"items": [[{"a": 1}, {"a": 1, "b": 2}]]}, {"items": [[{"a": 1, "b": 2}, {"a": 1}]]}, 1.0),
            ([], [], 1.0),
        ],
    )
    async def test_nested_duplicates_and_impossible_assignments(
        self, reference: Any, response: Any, expected: float
    ) -> None:
        """Keep multiplicity, nested matching and genuine mismatch behavior."""
        result = await JsonMatchGrader(strict_order=False, ignore_extra_keys=True).aevaluate(
            reference_response=json.dumps(reference), response=json.dumps(response)
        )
        assert result.score == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strict_order,ignore_extra_keys,expected", [(True, True, 0.0), (False, False, 0.0), (False, True, 1.0)]
    )
    async def test_order_and_extra_key_flags(
        self, strict_order: bool, ignore_extra_keys: bool, expected: float
    ) -> None:
        """Only explicitly unordered subset matching permits reassignment."""
        result = await JsonMatchGrader(strict_order=strict_order, ignore_extra_keys=ignore_extra_keys).aevaluate(
            reference_response='[{"a":1},{"a":1,"b":2}]',
            response='[{"a":1,"b":2,"extra":3},{"a":1}]',
        )
        assert result.score == expected
