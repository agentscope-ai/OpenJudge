"""
F1 Score Metric

Token-based F1 score calculation, following OpenAI Evals implementation.
Restructured to work with Grader framework.
"""

from collections import Counter

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class F1ScoreGrader(Grader):
    """
    Token-based F1 Score Grader

    Calculates F1 score based on token overlap between candidate and reference texts.
    Based on OpenAI Evals fuzzy_match implementation.

    The grader:
    1. Normalizes and tokenizes both texts
    2. Calculates token overlap using Counter
    3. Computes precision, recall, and F1 score

    Attributes:
        name: Grader name
        normalize: Whether to normalize text before tokenization

    Example:
        >>> grader = F1ScoreGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat is on the mat",
        ...     candidate="cat on mat"
        ... )
        >>> print(f"F1 Score: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "f1_score",
        normalize: bool = True,
        description: str = "Token-based F1 score metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        """Simple text normalization"""
        if not self.normalize:
            return text
        # Basic normalization: lowercase and strip
        return text.lower().strip()

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute F1 score

        Returns:
            tuple[float, dict]: (f1_score, details)
        """
        # Normalize and tokenize
        if self.normalize:
            candidate_norm = self._normalize_text(candidate)
            reference_norm = self._normalize_text(reference)
        else:
            candidate_norm = candidate
            reference_norm = reference

        # Tokenize by splitting on whitespace
        candidate_tokens = candidate_norm.split()
        reference_tokens = reference_norm.split()

        # Handle empty cases
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            if len(candidate_tokens) == 0 and len(reference_tokens) == 0:
                return 1.0, {
                    "precision": 1.0,
                    "recall": 1.0,
                }  # Both empty - perfect match
            else:
                return 0.0, {"precision": 0.0, "recall": 0.0}  # One empty - no match

        # Calculate token overlap using Counter
        candidate_counter = Counter(candidate_tokens)
        reference_counter = Counter(reference_tokens)
        common = candidate_counter & reference_counter
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, {"precision": 0.0, "recall": 0.0}

        # Calculate precision and recall
        precision = 1.0 * num_same / len(candidate_tokens)
        recall = 1.0 * num_same / len(reference_tokens)

        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall)

        details = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "common_tokens": num_same,
        }

        return f1, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate F1 score"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"F1 score: {score:.4f} (P={details['precision']:.3f}, R={details['recall']:.3f})",
            metadata=details,
        )


class TokenF1Grader(F1ScoreGrader):
    """
    Alias for F1ScoreGrader

    Provides a more descriptive name emphasizing token-based calculation.
    """

    def __init__(
        self,
        name: str = "token_f1",
        normalize: bool = True,
        description: str = "Token-based F1 score metric",
    ):
        super().__init__(name=name, normalize=normalize, description=description)


__all__ = [
    "F1ScoreGrader",
    "TokenF1Grader",
]
