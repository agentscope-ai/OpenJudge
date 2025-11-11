"""
Unit Tests for JSON Matching and Validation Graders

Tests for JsonMatchGrader and JsonValidatorGrader.
"""

import json

import pytest

from rm_gallery.core.metrics.format_check.json_match import (
    JsonMatchGrader,
    JsonValidatorGrader,
)


class TestJsonMatchGrader:
    """Test JsonMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match_simple(self):
        """Test exact match for simple JSON objects"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"name": "Alice", "age": 30}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_exact_match_different_order(self):
        """Test that dict key order doesn't matter"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"age": 30, "name": "Alice"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_no_match_different_values(self):
        """Test no match when values differ"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"name": "Bob", "age": 30}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_no_match_missing_key(self):
        """Test no match when key is missing"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": "Alice", "age": 30}', candidate='{"name": "Alice"}'
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_ignore_extra_keys(self):
        """Test ignore_extra_keys option"""
        grader = JsonMatchGrader(ignore_extra_keys=True)

        result = await grader.evaluate(
            reference='{"name": "Alice"}',
            candidate='{"name": "Alice", "age": 30, "city": "NYC"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_match_same_order(self):
        """Test list matching with same order"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(reference="[1, 2, 3]", candidate="[1, 2, 3]")

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_no_match_different_order(self):
        """Test list doesn't match with different order (strict_order=True)"""
        grader = JsonMatchGrader(strict_order=True)

        result = await grader.evaluate(reference="[1, 2, 3]", candidate="[3, 2, 1]")

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_list_match_different_order_allowed(self):
        """Test list matches with different order when strict_order=False"""
        grader = JsonMatchGrader(strict_order=False)

        result = await grader.evaluate(reference="[1, 2, 3]", candidate="[3, 2, 1]")

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_list_no_match_different_length(self):
        """Test lists with different lengths don't match"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(reference="[1, 2, 3]", candidate="[1, 2]")

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
            }
        )

        candidate = json.dumps(
            {
                "user": {
                    "name": "Alice",
                    "contacts": {"email": "alice@example.com", "phone": "123-4567"},
                },
                "active": True,
            }
        )

        result = await grader.evaluate(reference=reference, candidate=candidate)

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_nested_list_in_dict(self):
        """Test nested lists in dict"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"items": [1, 2, 3], "name": "test"}',
            candidate='{"items": [1, 2, 3], "name": "test"}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_invalid_candidate_json(self):
        """Test handling of invalid candidate JSON"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": "Alice"}', candidate="not valid json"
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False
        assert result.metadata["error"] == "candidate_parse_error"

    @pytest.mark.asyncio
    async def test_invalid_reference_json(self):
        """Test handling of invalid reference JSON"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference="not valid json", candidate='{"name": "Alice"}'
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False
        assert result.metadata["error"] == "reference_parse_error"

    @pytest.mark.asyncio
    async def test_null_values(self):
        """Test handling of null values"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"name": null}', candidate='{"name": null}'
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_boolean_values(self):
        """Test boolean value matching"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"active": true, "deleted": false}',
            candidate='{"active": true, "deleted": false}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_number_types(self):
        """Test different number types"""
        grader = JsonMatchGrader()

        result = await grader.evaluate(
            reference='{"int": 42, "float": 3.14}',
            candidate='{"int": 42, "float": 3.14}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_empty_structures(self):
        """Test empty dict and list"""
        grader = JsonMatchGrader()

        # Empty dict
        result = await grader.evaluate(reference="{}", candidate="{}")
        assert result.score == 1.0

        # Empty list
        result = await grader.evaluate(reference="[]", candidate="[]")
        assert result.score == 1.0


class TestJsonValidatorGrader:
    """Test JsonValidatorGrader"""

    @pytest.mark.asyncio
    async def test_valid_json_object(self):
        """Test valid JSON object"""
        grader = JsonValidatorGrader()

        result = await grader.evaluate(
            reference="", candidate='{"name": "Alice", "age": 30}'  # Not used
        )

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_valid_json_array(self):
        """Test valid JSON array"""
        grader = JsonValidatorGrader()

        result = await grader.evaluate(reference="", candidate='[1, 2, 3, "test"]')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_valid_json_primitives(self):
        """Test valid JSON primitives"""
        grader = JsonValidatorGrader()

        # String
        result = await grader.evaluate(reference="", candidate='"hello"')
        assert result.score == 1.0

        # Number
        result = await grader.evaluate(reference="", candidate="42")
        assert result.score == 1.0

        # Boolean
        result = await grader.evaluate(reference="", candidate="true")
        assert result.score == 1.0

        # Null
        result = await grader.evaluate(reference="", candidate="null")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_json_malformed(self):
        """Test invalid JSON (malformed)"""
        grader = JsonValidatorGrader()

        result = await grader.evaluate(
            reference="", candidate='{"name": "Alice"'  # Missing closing brace
        )

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False
        assert "error_message" in result.metadata

    @pytest.mark.asyncio
    async def test_invalid_json_not_json(self):
        """Test invalid JSON (not JSON at all)"""
        grader = JsonValidatorGrader()

        result = await grader.evaluate(
            reference="", candidate="This is just plain text"
        )

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False

    @pytest.mark.asyncio
    async def test_empty_string(self):
        """Test empty string is invalid JSON"""
        grader = JsonValidatorGrader()

        result = await grader.evaluate(reference="", candidate="")

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
