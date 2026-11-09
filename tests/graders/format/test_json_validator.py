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
    async def test_reject_non_rfc8259_literals_top_level(self):
        """NaN/Infinity/-Infinity are not valid JSON per RFC 8259 (#225)"""
        grader = JsonValidatorGrader()

        for text in ["NaN", "Infinity", "-Infinity"]:
            result = await grader.aevaluate(
                reference_response="", response=text
            )
            assert result.score == 0.0, f"Expected 0.0 for {text!r}, got {result.score}"
            assert result.metadata["is_valid"] is False
            assert "Non-RFC-8259" in result.metadata.get("error_message", "")

    @pytest.mark.asyncio
    async def test_reject_non_rfc8259_literals_nested(self):
        """Nested NaN/Infinity values are also rejected (#225)"""
        grader = JsonValidatorGrader()

        cases = ['{"a": NaN}', "[1, Infinity]", '{"k": -Infinity}']
        for text in cases:
            result = await grader.aevaluate(
                reference_response="", response=text
            )
            assert result.score == 0.0, f"Expected 0.0 for {text!r}"
            assert "Non-RFC-8259" in result.metadata.get("error_message", "")

    @pytest.mark.asyncio
    async def test_accept_normal_numbers_and_scientific_notation(self):
        """Ordinary numbers including scientific notation remain valid (#225)"""
        grader = JsonValidatorGrader()

        valid_numbers = ["1.5e10", "-0.0", "1e-7", "3.14159", "-42"]
        for text in valid_numbers:
            result = await grader.aevaluate(
                reference_response="", response=text
            )
            assert result.score == 1.0, f"Expected 1.0 for {text!r}"

    @pytest.mark.asyncio
    async def test_nan_in_string_value_is_valid(self):
        """The literal 'nan' inside a JSON string is valid — only bare literals are rejected"""
        grader = JsonValidatorGrader()

        # 'nan' as a string value is perfectly valid JSON
        result = await grader.aevaluate(
            reference_response="", response='{"key": "nan"}'
        )
        assert result.score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
