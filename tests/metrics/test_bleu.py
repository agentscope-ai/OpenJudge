"""
Unit Tests for BLEU Grader

Test BLEU score calculation for machine translation evaluation.
"""

import pytest

from rm_gallery.core.metrics.nlp_metrics.bleu import BLEUGrader, SentenceBLEUGrader


class TestBLEUBasic:
    """Basic BLEU functionality tests"""

    @pytest.mark.asyncio
    async def test_perfect_match(self):
        """Test perfect match returns score of 1.0"""
        grader = BLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        assert result.score == 1.0
        assert "precisions" in result.metadata

    @pytest.mark.asyncio
    async def test_complete_mismatch(self):
        """Test completely different sentences"""
        grader = BLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat",
            candidate="hello world foo bar baz qux",
        )

        assert result.score < 0.1  # Very low score for completely different text

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching"""
        grader = BLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the dog is on the mat"
        )

        # Should have some overlap but not perfect
        assert 0.3 < result.score < 0.9

    @pytest.mark.asyncio
    async def test_word_order_matters(self):
        """Test that word order affects BLEU score"""
        grader = BLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the mat on is cat the"
        )

        # Same words but different order should give lower score
        assert result.score < 1.0


class TestBLEUParameters:
    """Test BLEU with different parameters"""

    @pytest.mark.asyncio
    async def test_different_max_ngram_orders(self):
        """Test different n-gram orders"""
        # Test with different max n-gram orders
        for n in [1, 2, 3, 4]:
            grader = BLEUGrader(max_ngram_order=n)
            result = await grader.evaluate(
                reference="the quick brown fox jumps over the lazy dog",
                candidate="the quick brown fox jumps over the lazy dog",
            )
            assert result.score == 1.0
            assert len(result.metadata["precisions"]) == n

    @pytest.mark.asyncio
    async def test_smoothing_methods(self):
        """Test different smoothing methods"""
        # Test different smoothing methods
        for method in ["none", "floor", "add-k", "exp"]:
            grader = BLEUGrader(smooth_method=method)
            result = await grader.evaluate(
                reference="the cat sat on the mat", candidate="the cat"
            )
            assert 0.0 <= result.score <= 1.0


class TestBLEUEdgeCases:
    """Test edge cases"""

    @pytest.mark.asyncio
    async def test_empty_candidate(self):
        """Test handling of empty candidate"""
        grader = BLEUGrader()
        result = await grader.evaluate(reference="the cat is on the mat", candidate="")

        # Empty candidate should give zero score
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        """Test handling of empty reference"""
        grader = BLEUGrader()
        result = await grader.evaluate(reference="", candidate="the cat")

        # Empty reference should give zero score
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_single_word_sentences(self):
        """Test single word sentences"""
        grader = BLEUGrader()
        result = await grader.evaluate(reference="cat", candidate="cat")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_very_long_sentences(self):
        """Test with very long sentences"""
        grader = BLEUGrader()
        long_sentence = " ".join(["word"] * 500)
        result = await grader.evaluate(reference=long_sentence, candidate=long_sentence)

        assert result.score == 1.0


class TestBLEUDetails:
    """Test BLEU result details"""

    @pytest.mark.asyncio
    async def test_precision_details(self):
        """Test that precision details are included"""
        grader = BLEUGrader(max_ngram_order=4)
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        assert "precisions" in result.metadata
        assert len(result.metadata["precisions"]) == 4
        # Perfect match should have all precisions = 1.0
        for prec in result.metadata["precisions"]:
            assert prec == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_brevity_penalty(self):
        """Test brevity penalty calculation"""
        grader = BLEUGrader()

        # Short candidate
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the cat"
        )

        assert "bp" in result.metadata
        # Brevity penalty should be < 1.0 for shorter candidate
        assert result.metadata["bp"] < 1.0

    @pytest.mark.asyncio
    async def test_length_information(self):
        """Test that length information is included"""
        grader = BLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the dog is on the rug"
        )

        assert "sys_len" in result.metadata  # System (candidate) length
        assert "ref_len" in result.metadata  # Reference length
        assert result.metadata["sys_len"] == result.metadata["ref_len"]


class TestSentenceBLEU:
    """Test sentence-level BLEU variant"""

    @pytest.mark.asyncio
    async def test_sentence_bleu_basic(self):
        """Test basic sentence BLEU"""
        grader = SentenceBLEUGrader()
        result = await grader.evaluate(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_sentence_bleu_vs_corpus_bleu(self):
        """Compare sentence-level and corpus-level BLEU"""
        sentence_grader = SentenceBLEUGrader()
        corpus_grader = BLEUGrader()

        sentence_result = await sentence_grader.evaluate(
            reference="the cat is on the mat", candidate="the dog is on the mat"
        )
        corpus_result = await corpus_grader.evaluate(
            reference="the cat is on the mat", candidate="the dog is on the mat"
        )

        # Scores may differ slightly due to different calculation methods
        assert 0.0 <= sentence_result.score <= 1.0
        assert 0.0 <= corpus_result.score <= 1.0
