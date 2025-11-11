"""
Fuzzy Match Metric

Fuzzy matching metrics based on Levenshtein Distance.
Restructured to work with Grader framework.
"""

import Levenshtein

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class FuzzyMatchGrader(Grader):
    """
    Fuzzy Match Grader

    Calculates text similarity using Levenshtein edit distance.
    Supports three matching modes:
    - ratio: Standard similarity ratio
    - partial_ratio: Partial string matching
    - token_sort_ratio: Token order-independent matching

    Attributes:
        name: Grader name
        method: Matching method
        threshold: Match threshold

    Example:
        >>> grader = FuzzyMatchGrader(method="ratio", threshold=0.8)
        >>> result = await grader.evaluate(
        ...     reference="hello world",
        ...     candidate="hello worl"
        ... )
        >>> print(f"Score: {result.score:.2f}, Matched: {result.metadata['matched']}")
    """

    def __init__(
        self,
        name: str = "fuzzy_match",
        method: str = "ratio",
        threshold: float = 0.8,
        description: str = "Fuzzy matching based on Levenshtein distance",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.method = method
        self.threshold = threshold

    def _partial_ratio(self, s1: str, s2: str) -> float:
        """Partial string matching"""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0 if s1 != s2 else 1.0

        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        m = len(shorter)
        max_ratio = 0.0

        for i in range(len(longer) - m + 1):
            ratio = Levenshtein.ratio(shorter, longer[i : i + m])
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def _token_sort_ratio(self, s1: str, s2: str) -> float:
        """Token order-independent fuzzy matching"""
        tokens1 = sorted(s1.split())
        tokens2 = sorted(s2.split())
        return Levenshtein.ratio(" ".join(tokens1), " ".join(tokens2))

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute fuzzy match score

        Returns:
            tuple[float, dict]: (score, details)
        """
        if self.method == "ratio":
            score = Levenshtein.ratio(candidate, reference)
        elif self.method == "partial_ratio":
            score = self._partial_ratio(candidate, reference)
        elif self.method == "token_sort_ratio":
            score = self._token_sort_ratio(candidate, reference)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'ratio', 'partial_ratio', or 'token_sort_ratio'"
            )

        matched = score >= self.threshold

        details = {
            "method": self.method,
            "threshold": self.threshold,
            "matched": matched,
        }

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate fuzzy match"""
        score, details = self._compute(reference, candidate)

        matched_text = "matched" if details["matched"] else "not matched"
        return GraderScore(
            score=score,
            reason=f"Fuzzy match ({self.method}): {score:.4f} ({matched_text})",
            metadata=details,
        )


class EditDistanceGrader(Grader):
    """
    Edit Distance Grader

    Returns normalized Levenshtein edit distance as a similarity score.

    Attributes:
        name: Grader name
        normalize_by_length: Whether to normalize by length

    Example:
        >>> grader = EditDistanceGrader()
        >>> result = await grader.evaluate(
        ...     reference="kitten",
        ...     candidate="sitting"
        ... )
        >>> print(f"Similarity: {result.score:.2f}")
    """

    def __init__(
        self,
        name: str = "edit_distance",
        normalize_by_length: bool = True,
        description: str = "Edit distance based similarity metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.normalize_by_length = normalize_by_length

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute edit distance"""
        raw_distance = Levenshtein.distance(candidate, reference)
        max_len = max(len(candidate), len(reference))

        # Normalize to similarity score
        if self.normalize_by_length and max_len > 0:
            normalized_score = 1.0 - (raw_distance / max_len)
        else:
            normalized_score = 1.0 / (1.0 + raw_distance)

        # Ensure within [0, 1] range
        normalized_score = max(0.0, min(1.0, normalized_score))

        details = {
            "raw_distance": raw_distance,
            "max_length": max_len,
            "normalize_by_length": self.normalize_by_length,
        }

        return normalized_score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate edit distance"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Edit distance similarity: {score:.4f} (distance={details['raw_distance']})",
            metadata=details,
        )


__all__ = [
    "FuzzyMatchGrader",
    "EditDistanceGrader",
]
