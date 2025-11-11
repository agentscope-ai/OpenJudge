"""
Unit Tests for New Grader Architecture

Tests for all Grader classes after Core class merging.
"""

import pytest

from rm_gallery.core.metrics.format_check.json_match import (
    JsonMatchGrader,
    JsonValidatorGrader,
)
from rm_gallery.core.metrics.nlp_metrics.bleu import BLEUGrader, SentenceBLEUGrader
from rm_gallery.core.metrics.nlp_metrics.gleu import ChrFGrader, GLEUGrader
from rm_gallery.core.metrics.nlp_metrics.meteor import METEORGrader
from rm_gallery.core.metrics.nlp_metrics.rouge import (
    ROUGE1Grader,
    ROUGE2Grader,
    ROUGEGrader,
    ROUGELGrader,
)
from rm_gallery.core.metrics.nlp_metrics.rouge_ngram import (
    ROUGE3Grader,
    ROUGE4Grader,
    ROUGE5Grader,
)
from rm_gallery.core.metrics.string_check.exact_match import (
    ExactMatchGrader,
    PrefixMatchGrader,
    RegexMatchGrader,
    SuffixMatchGrader,
)
from rm_gallery.core.metrics.string_check.substring import (
    CharacterOverlapGrader,
    ContainsAllGrader,
    ContainsAnyGrader,
    SubstringMatchGrader,
    WordOverlapGrader,
)
from rm_gallery.core.metrics.text_similarity.cosine import (
    CosineSimilarityGrader,
    JaccardSimilarityGrader,
)
from rm_gallery.core.metrics.text_similarity.f1_score import (
    F1ScoreGrader,
    TokenF1Grader,
)
from rm_gallery.core.metrics.text_similarity.fuzzy import (
    EditDistanceGrader,
    FuzzyMatchGrader,
)


class TestFuzzyMatchGrader:
    """Test FuzzyMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect score"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(reference="hello world", candidate="hello world")

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching"""
        grader = FuzzyMatchGrader()
        result = await grader.evaluate(
            reference="hello world", candidate="hello worl"  # Missing 'd'
        )

        assert 0.9 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_different_methods(self):
        """Test different fuzzy matching methods"""
        # Ratio method
        grader_ratio = FuzzyMatchGrader(method="ratio")
        result = await grader_ratio.evaluate(
            reference="the quick brown fox", candidate="the quick brown fox"
        )
        assert result.score == 1.0
        assert result.metadata["method"] == "ratio"

        # Token sort ratio
        grader_token = FuzzyMatchGrader(method="token_sort_ratio")
        result = await grader_token.evaluate(
            reference="brown quick the fox", candidate="the quick brown fox"
        )
        assert result.score == 1.0


class TestEditDistanceGrader:
    """Test EditDistanceGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match"""
        grader = EditDistanceGrader()
        result = await grader.evaluate(reference="hello", candidate="hello")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_char_difference(self):
        """Test one character difference"""
        grader = EditDistanceGrader()
        result = await grader.evaluate(reference="hello", candidate="helo")

        assert 0.7 < result.score < 1.0
        assert result.metadata["raw_distance"] == 1


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
    async def test_different_key_order(self):
        """Test that dict key order doesn't matter"""
        grader = JsonMatchGrader()
        result = await grader.evaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"age": 30, "name": "Alice"}',
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test no match when values differ"""
        grader = JsonMatchGrader()
        result = await grader.evaluate(
            reference='{"name": "Alice"}', candidate='{"name": "Bob"}'
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON"""
        grader = JsonMatchGrader()
        result = await grader.evaluate(
            reference='{"name": "Alice"}', candidate="not valid json"
        )

        assert result.score == 0.0
        assert "error" in result.metadata


class TestJsonValidatorGrader:
    """Test JsonValidatorGrader"""

    @pytest.mark.asyncio
    async def test_valid_json(self):
        """Test valid JSON object"""
        grader = JsonValidatorGrader()
        result = await grader.evaluate(candidate='{"name": "Alice", "age": 30}')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test invalid JSON (malformed)"""
        grader = JsonValidatorGrader()
        result = await grader.evaluate(candidate='{"name": "Alice"')  # Missing }

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False

    @pytest.mark.asyncio
    async def test_valid_json_array(self):
        """Test valid JSON array"""
        grader = JsonValidatorGrader()
        result = await grader.evaluate(candidate='[1, 2, 3, "test"]')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True


class TestExactMatchGrader:
    """Test ExactMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match"""
        grader = ExactMatchGrader()
        result = await grader.evaluate(reference="hello", candidate="hello")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case insensitive matching"""
        grader = ExactMatchGrader(case_sensitive=False)
        result = await grader.evaluate(reference="Hello", candidate="hello")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test no match"""
        grader = ExactMatchGrader()
        result = await grader.evaluate(reference="hello", candidate="world")

        assert result.score == 0.0


class TestSubstringMatchGrader:
    """Test SubstringMatchGrader"""

    @pytest.mark.asyncio
    async def test_substring_found(self):
        """Test substring found"""
        grader = SubstringMatchGrader()
        result = await grader.evaluate(
            reference="cat", candidate="The cat sat on the mat"
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_substring_not_found(self):
        """Test substring not found"""
        grader = SubstringMatchGrader()
        result = await grader.evaluate(
            reference="dog", candidate="The cat sat on the mat"
        )

        assert result.score == 0.0


class TestCosineSimilarityGrader:
    """Test CosineSimilarityGrader"""

    @pytest.mark.asyncio
    async def test_identical_texts(self):
        """Test identical texts have similarity 1.0"""
        grader = CosineSimilarityGrader()
        result = await grader.evaluate(
            reference="the cat sat on the mat", candidate="the cat sat on the mat"
        )

        assert result.score > 0.99

    @pytest.mark.asyncio
    async def test_similar_texts(self):
        """Test similar texts have high similarity"""
        grader = CosineSimilarityGrader()
        result = await grader.evaluate(
            reference="the cat sat on the mat", candidate="the dog sat on the mat"
        )

        assert 0.5 < result.score < 1.0


class TestJaccardSimilarityGrader:
    """Test JaccardSimilarityGrader"""

    @pytest.mark.asyncio
    async def test_identical_texts(self):
        """Test identical texts"""
        grader = JaccardSimilarityGrader()
        result = await grader.evaluate(reference="hello world", candidate="hello world")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        """Test no overlap"""
        grader = JaccardSimilarityGrader()
        result = await grader.evaluate(
            reference="hello world", candidate="goodbye universe"
        )

        assert result.score == 0.0


class TestAllGraderBasics:
    """Test that all graders work with basic inputs"""

    @pytest.mark.asyncio
    async def test_all_graders_can_evaluate(self):
        """Test that all graders can be instantiated and evaluate"""
        graders = [
            BLEUGrader(),
            SentenceBLEUGrader(),
            GLEUGrader(),
            ChrFGrader(),
            METEORGrader(),
            ROUGEGrader(),
            ROUGE1Grader(),
            ROUGE2Grader(),
            ROUGELGrader(),
            ROUGE3Grader(),
            ROUGE4Grader(),
            ROUGE5Grader(),
            F1ScoreGrader(),
            TokenF1Grader(),
            FuzzyMatchGrader(),
            EditDistanceGrader(),
            CosineSimilarityGrader(),
            JaccardSimilarityGrader(),
            ExactMatchGrader(),
            PrefixMatchGrader(),
            SuffixMatchGrader(),
            RegexMatchGrader(),
            SubstringMatchGrader(),
            ContainsAllGrader(substrings=["test"]),
            ContainsAnyGrader(substrings=["test"]),
            WordOverlapGrader(),
            CharacterOverlapGrader(),
            JsonMatchGrader(),
        ]

        reference = "the cat sat on the mat"
        candidate = "the cat sat on the mat"

        for grader in graders:
            result = await grader.evaluate(reference=reference, candidate=candidate)
            assert hasattr(result, "score")
            assert hasattr(result, "reason")
            assert hasattr(result, "metadata")
            assert 0.0 <= result.score <= 1.0
            print(f"✓ {grader.name}: score={result.score:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
