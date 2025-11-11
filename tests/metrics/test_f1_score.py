"""
Unit Tests for F1 Score Grader

Test token-based F1 score calculation following OpenAI Evals implementation.
"""

import pytest

from rm_gallery.core.metrics.text_similarity.f1_score import (
    F1ScoreGrader,
    TokenF1Grader,
)


class TestF1ScoreBasic:
    """Basic F1 score functionality tests"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect F1 score"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(reference="hello world", candidate="hello world")

        assert result.score == 1.0
        assert result.metadata["precision"] == 1.0
        assert result.metadata["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        """Test completely different strings return 0 F1 score"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(
            reference="hello world", candidate="goodbye universe"
        )

        assert result.score == 0.0
        assert result.metadata["precision"] == 0.0
        assert result.metadata["recall"] == 0.0

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        """Test partial token overlap"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="cat on mat"
        )

        # Tokens: reference has more tokens, candidate has subset
        # The exact score depends on normalization
        assert 0.5 < result.score < 1.0
        assert result.metadata["precision"] >= 0.5
        assert result.metadata["recall"] >= 0.5

    @pytest.mark.asyncio
    async def test_word_order_matters(self):
        """Test that word order doesn't affect F1 (token-based)"""
        grader = F1ScoreGrader()

        result1 = await grader.evaluate(
            reference="the quick brown fox", candidate="fox brown quick the"
        )
        result2 = await grader.evaluate(
            reference="the quick brown fox", candidate="the quick brown fox"
        )

        # Both should have similar F1 (same tokens, may differ due to normalization)
        assert abs(result1.score - result2.score) < 0.1


class TestF1ScoreNormalization:
    """Test normalization effects on F1 score"""

    @pytest.mark.asyncio
    async def test_with_normalization(self):
        """Test with normalization enabled"""
        grader = F1ScoreGrader(normalize=True)
        result = await grader.evaluate(reference="Hello World", candidate="hello world")

        # With normalization, case differences shouldn't matter
        assert result.score > 0.9

    @pytest.mark.asyncio
    async def test_without_normalization(self):
        """Test that disabling normalization preserves case"""
        grader = F1ScoreGrader(normalize=False)
        result = await grader.evaluate(reference="Hello World", candidate="hello world")

        # Without normalization, case differences matter
        assert result.score < 1.0


class TestF1ScoreEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_strings(self):
        """Test handling of empty strings"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(reference="", candidate="")

        # Both empty - perfect match
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        """Test empty reference with non-empty candidate"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(reference="", candidate="hello")

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_empty_candidate(self):
        """Test non-empty reference with empty candidate"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(reference="hello", candidate="")

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_single_token(self):
        """Test single token matching"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(reference="hello", candidate="hello")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_duplicate_tokens(self):
        """Test handling of duplicate tokens"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(
            reference="hello hello world", candidate="hello world world"
        )

        # Should handle token counts correctly
        assert 0.5 < result.score < 0.8


class TestF1ScorePrecisionRecall:
    """Test precision and recall calculations"""

    @pytest.mark.asyncio
    async def test_high_precision_low_recall(self):
        """Test case with high precision but low recall"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(
            reference="the quick brown fox jumps over the lazy dog",
            candidate="quick brown fox",
        )

        # All candidate tokens should be in reference (high precision)
        # But only fraction of reference tokens in candidate (low recall)
        assert result.metadata["precision"] > result.metadata["recall"]

    @pytest.mark.asyncio
    async def test_low_precision_high_recall(self):
        """Test case with low precision but high recall"""
        grader = F1ScoreGrader()
        result = await grader.evaluate(
            reference="quick brown fox",
            candidate="the quick brown fox jumps over the lazy dog",
        )

        # All reference tokens should be in candidate (high recall)
        # But many extra candidate tokens (low precision)
        assert result.metadata["recall"] > result.metadata["precision"]


class TestTokenF1Alias:
    """Test TokenF1Grader alias"""

    @pytest.mark.asyncio
    async def test_alias_works(self):
        """Test that TokenF1Grader works"""
        grader = TokenF1Grader()
        result = await grader.evaluate(reference="hello world", candidate="hello world")

        assert result.score == 1.0
