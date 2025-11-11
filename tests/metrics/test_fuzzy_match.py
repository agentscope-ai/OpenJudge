"""
Unit Tests for Fuzzy Match Grader

Test fuzzy matching functionality including exact match, partial match, and token sorting.
"""

import pytest

from rm_gallery.core.metrics.text_similarity.fuzzy import FuzzyMatchGrader


class TestFuzzyMatchBasic:
    """Basic fuzzy match functionality tests"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect score"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="hello world", candidate="hello world")

        assert result.score == 1.0
        assert "matched" in result.metadata
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_complete_mismatch(self):
        """Test completely different strings return low score"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(
            reference="hello world", candidate="goodbye universe"
        )

        assert 0.0 <= result.score < 0.5
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(
            reference="hello world", candidate="hello worl"  # Missing 'd'
        )

        assert 0.9 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case insensitive matching"""
        # Note: FuzzyMatchGrader doesn't have normalize_text parameter
        # Case sensitivity is handled by the fuzzy matching algorithm
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="Hello World", candidate="hello world")

        # Fuzzy match is case-sensitive, so won't be 1.0, but should be high
        assert result.score > 0.8


class TestFuzzyMatchMethods:
    """Test different fuzzy matching methods"""

    @pytest.mark.asyncio
    async def test_ratio_method(self):
        """Test standard ratio method"""
        grader = FuzzyMatchGrader(method="ratio")
        result = await grader.evaluate(
            reference="the quick brown fox", candidate="the quick brown fox"
        )

        assert result.score == 1.0
        assert result.metadata["method"] == "ratio"

    @pytest.mark.asyncio
    async def test_partial_ratio_method(self):
        """Test partial ratio for substring matching"""
        grader = FuzzyMatchGrader(method="partial_ratio")
        result = await grader.evaluate(
            reference="the quick brown fox jumps", candidate="quick brown fox"
        )

        # Partial ratio should give high score for substring match
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_token_sort_ratio_method(self):
        """Test token sort ratio for order-independent matching"""
        grader = FuzzyMatchGrader(method="token_sort_ratio")
        result = await grader.evaluate(
            reference="brown quick the fox", candidate="the quick brown fox"
        )

        # Token sort should give perfect score for same words different order
        assert result.score == 1.0


class TestFuzzyMatchMultipleReferences:
    """Test fuzzy matching with multiple reference texts"""

    @pytest.mark.asyncio
    async def test_multiple_references_exact_match(self):
        """Test exact match with multiple references"""
        # Note: Grader architecture evaluates single reference at a time
        # For multiple references, we evaluate each separately and take the best
        grader = FuzzyMatchGrader()

        references = ["hello world", "hi there", "greetings"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.evaluate(reference=ref, candidate=candidate)
            scores.append(result.score)

        assert max(scores) == 1.0
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_multiple_references_best_match(self):
        """Test that best matching reference is selected"""
        grader = FuzzyMatchGrader()

        references = ["completely different text", "hello world", "another text"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.evaluate(reference=ref, candidate=candidate)
            scores.append(result.score)

        # Should match the second reference perfectly
        assert max(scores) == 1.0

    @pytest.mark.asyncio
    async def test_multiple_references_partial_match(self):
        """Test partial matching with multiple references"""
        grader = FuzzyMatchGrader()

        references = ["hello", "world", "foo bar"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.evaluate(reference=ref, candidate=candidate)
            scores.append(result.score)

        # Should have some match
        assert max(scores) > 0.0


class TestFuzzyMatchEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_strings(self):
        """Test handling of empty strings"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="", candidate="")

        # Empty strings should match perfectly
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference_non_empty_candidate(self):
        """Test empty reference with non-empty candidate"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="", candidate="hello")

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_very_long_strings(self):
        """Test performance with long strings"""
        grader = FuzzyMatchGrader()
        long_text = "word " * 1000  # 1000 words
        result = await grader.evaluate(reference=long_text, candidate=long_text)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(
            reference="hello@world#2024!", candidate="hello@world#2024!"
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_unicode_characters(self):
        """Test handling of Unicode characters"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="你好世界", candidate="你好世界")

        assert result.score == 1.0


class TestFuzzyMatchThreshold:
    """Test threshold functionality"""

    @pytest.mark.asyncio
    async def test_threshold_matching(self):
        """Test threshold-based matching decision"""
        grader = FuzzyMatchGrader(threshold=0.9)

        # High similarity - should match
        result = await grader.evaluate(
            reference="the quick brown fox", candidate="the quick brown fo"
        )

        if result.score >= 0.9:
            assert result.metadata["matched"] is True
        else:
            assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_different_thresholds(self):
        """Test different threshold values"""

        # Strict threshold
        grader_strict = FuzzyMatchGrader(threshold=0.99)
        result_strict = await grader_strict.evaluate(
            reference="hello world", candidate="hello worl"
        )

        # Lenient threshold
        grader_lenient = FuzzyMatchGrader(threshold=0.80)
        result_lenient = await grader_lenient.evaluate(
            reference="hello world", candidate="hello worl"
        )

        # Scores should be the same, but matching decision may differ
        assert result_strict.score == result_lenient.score
