# -*- coding: utf-8 -*-
"""
Tests for StringMatchGrader
"""

import pytest

from openjudge.graders.text.string_match import StringMatchGrader

# pylint: disable=too-many-public-methods


@pytest.mark.asyncio
class TestStringMatchGrader:
    """Test cases for unified StringMatchGrader"""

    async def test_exact_match_case_sensitive(self):
        """Test exact match with case sensitivity"""
        grader = StringMatchGrader(algorithm="exact_match", case_sensitive=True)
        result = await grader.aevaluate(
            reference_response="Hello World",
            response="Hello World",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_exact_match_case_insensitive(self):
        """Test exact match without case sensitivity"""
        grader = StringMatchGrader(algorithm="exact_match", case_sensitive=False)
        result = await grader.aevaluate(
            reference_response="Hello World",
            response="hello world",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_exact_match_ignore_whitespace(self):
        """Test exact match ignoring whitespace"""
        grader = StringMatchGrader(algorithm="exact_match", ignore_whitespace=True)
        result = await grader.aevaluate(
            reference_response="Hello World",
            response="HelloWorld",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_prefix_match_success(self):
        """Test prefix match success"""
        grader = StringMatchGrader(algorithm="prefix_match")
        result = await grader.aevaluate(
            reference_response="Hello",
            response="Hello World",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_prefix_match_failure(self):
        """Test prefix match failure"""
        grader = StringMatchGrader(algorithm="prefix_match")
        result = await grader.aevaluate(
            reference_response="World",
            response="Hello World",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_suffix_match_success(self):
        """Test suffix match success"""
        grader = StringMatchGrader(algorithm="suffix_match")
        result = await grader.aevaluate(
            reference_response="World",
            response="Hello World",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_suffix_match_failure(self):
        """Test suffix match failure"""
        grader = StringMatchGrader(algorithm="suffix_match")
        result = await grader.aevaluate(
            reference_response="Hello",
            response="Hello World",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_regex_match_success(self):
        """Test regex match success"""
        grader = StringMatchGrader(algorithm="regex_match")
        result = await grader.aevaluate(
            reference_response=r"\d{3}-\d{4}",
            response="My phone is 123-4567",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_regex_match_with_pattern_param(self):
        """Test regex match with pattern parameter"""
        grader = StringMatchGrader(algorithm="regex_match")
        result = await grader.aevaluate(
            reference_response="",
            response="test@example.com",
            pattern=r"[\w.-]+@[\w.-]+\.\w+",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_regex_match_invalid_pattern(self):
        """Test regex match with invalid pattern"""
        grader = StringMatchGrader(algorithm="regex_match")
        result = await grader.aevaluate(
            reference_response="[invalid(",
            response="test",
        )
        assert result.score == 0.0
        assert "error" in result.metadata

    async def test_substring_match_success(self):
        """Test substring match success"""
        grader = StringMatchGrader(algorithm="substring_match")
        result = await grader.aevaluate(
            reference_response="cat",
            response="The cat sat on the mat",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_substring_match_failure(self):
        """Test substring match failure"""
        grader = StringMatchGrader(algorithm="substring_match")
        result = await grader.aevaluate(
            reference_response="dog",
            response="The cat sat on the mat",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_substring_match_bidirectional(self):
        """Test substring match bidirectional"""
        grader = StringMatchGrader(algorithm="substring_match")
        result = await grader.aevaluate(
            reference_response="The cat sat on the mat",
            response="cat",
            bidirectional=True,
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_contains_all_success(self):
        """Test contains all success"""
        grader = StringMatchGrader(algorithm="contains_all")
        result = await grader.aevaluate(
            reference_response="",
            response="The cat sat on the mat",
            substrings=["cat", "mat"],
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True
        assert len(result.metadata["missing_substrings"]) == 0

    async def test_contains_all_partial(self):
        """Test contains all partial match"""
        grader = StringMatchGrader(algorithm="contains_all")
        result = await grader.aevaluate(
            reference_response="",
            response="The cat sat on the mat",
            substrings=["cat", "dog", "mat"],
        )
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.metadata["matched"] is False
        assert "dog" in result.metadata["missing_substrings"]

    async def test_contains_any_success(self):
        """Test contains any success"""
        grader = StringMatchGrader(algorithm="contains_any")
        result = await grader.aevaluate(
            reference_response="",
            response="The cat sat on the mat",
            substrings=["dog", "cat"],
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True
        assert "cat" in result.metadata["matched_substrings"]

    async def test_contains_any_failure(self):
        """Test contains any failure"""
        grader = StringMatchGrader(algorithm="contains_any")
        result = await grader.aevaluate(
            reference_response="",
            response="The cat sat on the mat",
            substrings=["dog", "bird"],
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_word_overlap(self):
        """Test word overlap"""
        grader = StringMatchGrader(algorithm="word_overlap")
        result = await grader.aevaluate(
            reference_response="the cat sat on the mat",
            response="the dog sat on the rug",
        )
        # Overlapping words: "the", "sat", "on" = 3 out of 5 unique words in reference_response
        # Reference has: {"the", "cat", "sat", "on", "mat"} = 5 unique words
        # Overlap: {"the", "sat", "on"} = 3 words
        assert result.score == pytest.approx(3.0 / 5.0)

    async def test_char_overlap(self):
        """Test character overlap"""
        grader = StringMatchGrader(algorithm="char_overlap")
        result = await grader.aevaluate(
            reference_response="hello",
            response="helo",
        )
        # All characters in "hello" {h, e, l, o} are in "helo"
        assert result.score == 1.0

    @pytest.mark.parametrize("algorithm", ["prefix_match", "substring_match", "contains_all"])
    async def test_empty_reference_does_not_match(self, algorithm):
        """Test that an empty reference does not produce a perfect score"""
        grader = StringMatchGrader(algorithm=algorithm)
        result = await grader.aevaluate(
            reference_response="",
            response="completely wrong answer",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_invalid_algorithm(self):
        """Test invalid algorithm"""
        with pytest.raises(ValueError) as exc_info:
            StringMatchGrader(algorithm="invalid_algorithm")
        assert "Unknown self.algorithm" in str(exc_info.value)

    async def test_algorithm_metadata(self):
        """Test that algorithm is included in metadata"""
        grader = StringMatchGrader(algorithm="exact_match")
        result = await grader.aevaluate(
            reference_response="test",
            response="test",
        )
        assert result.metadata["algorithm"] == "exact_match"
