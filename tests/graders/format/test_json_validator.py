import pytest

from openjudge.graders.format.json.json_validator import JsonValidatorGrader


class TestJsonValidatorGrader:
    """Test JsonValidatorGrader"""

    @pytest.mark.asyncio
    async def test_valid_json_object(self):
        """Test valid JSON object"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(
            reference_response="",
            response='{"name": "Alice", "age": 30}',  # Not used
        )

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_valid_json_array(self):
        """Test valid JSON array"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(reference_response="", response='[1, 2, 3, "test"]')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_valid_json_primitives(self):
        """Test valid JSON primitives"""
        grader = JsonValidatorGrader()

        # String
        result = await grader.aevaluate(reference_response="", response='"hello"')
        assert result.score == 1.0

        # Number
        result = await grader.aevaluate(reference_response="", response="42")
        assert result.score == 1.0

        # Boolean
        result = await grader.aevaluate(reference_response="", response="true")
        assert result.score == 1.0

        # Null
        result = await grader.aevaluate(reference_response="", response="null")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_json_malformed(self):
        """Test invalid JSON (malformed)"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(
            reference_response="",
            response='{"name": "Alice"',  # Missing closing brace
        )

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False
        assert "error_message" in result.metadata

    @pytest.mark.asyncio
    async def test_invalid_json_not_json(self):
        """Test invalid JSON (not JSON at all)"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(
            reference_response="",
            response="This is just plain text",
        )

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False

    @pytest.mark.asyncio
    async def test_empty_string(self):
        """Test empty string is invalid JSON"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(reference_response="", response="")

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        ["NaN", "Infinity", "-Infinity", '{"a": NaN}', "[1, Infinity]", '{"a": -Infinity}'],
    )
    async def test_non_finite_literals_are_not_valid_json(self, response: str):
        """Python's json module accepts NaN/Infinity, but JSON itself does not"""
        grader = JsonValidatorGrader()

        result = await grader.aevaluate(reference_response="", response=response)

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False
        assert "Invalid JSON number literal" in result.metadata["error_message"]

    @pytest.mark.asyncio
    async def test_finite_numbers_and_null_still_valid(self):
        """Rejecting non-finite literals must not affect ordinary values"""
        grader = JsonValidatorGrader()

        for response in ['{"a": 1.5e10}', '{"a": -0.0}', '{"a": null}', "[0, 1e-7]"]:
            result = await grader.aevaluate(reference_response="", response=response)

            assert result.score == 1.0
            assert result.metadata["is_valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
