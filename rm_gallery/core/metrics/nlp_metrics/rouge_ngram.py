"""
Custom ROUGE N-gram Metrics

Custom implementation of ROUGE-3, ROUGE-4, and ROUGE-5 metrics.
Restructured to work with Grader framework.
"""

from collections import Counter
from typing import List, Tuple

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class ROUGENGramGrader(Grader):
    """
    ROUGE N-gram Grader

    ROUGE-N measures n-gram overlap between candidate and reference texts.
    It calculates precision, recall, and F1-score based on n-gram matching.

    Attributes:
        name: Grader name
        n: N-gram size
        score_type: Which score to return (precision/recall/fmeasure)

    Example:
        >>> grader = ROUGENGramGrader(n=3)
        >>> result = await grader.evaluate(
        ...     reference="the quick brown fox jumps",
        ...     candidate="the quick brown fox jumps"
        ... )
        >>> print(f"ROUGE-{grader.n}: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "rouge_ngram",
        n: int = 3,
        score_type: str = "fmeasure",
        description: str = "ROUGE N-gram overlap metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.n = n
        self.score_type = score_type

    def _get_ngrams(self, text: str, n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from text"""
        tokens = text.split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.append(ngram)
        return ngrams

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute ROUGE N-gram score

        Returns:
            tuple[float, dict]: (score, details)
        """
        # Get n-grams
        ref_ngrams = self._get_ngrams(reference, self.n)
        cand_ngrams = self._get_ngrams(candidate, self.n)

        # Count matches
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)

        # Calculate overlap
        overlap = cand_counter & ref_counter
        overlap_count = sum(overlap.values())

        # Calculate total counts
        ref_count = sum(ref_counter.values())
        cand_count = sum(cand_counter.values())

        # Calculate precision, recall, and F1
        if cand_count == 0:
            precision = 0.0
        else:
            precision = overlap_count / cand_count

        if ref_count == 0:
            recall = 0.0
        else:
            recall = overlap_count / ref_count

        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)

        # Select score based on score_type
        if self.score_type == "precision":
            score = precision
        elif self.score_type == "recall":
            score = recall
        else:  # fmeasure (default)
            score = fmeasure

        details = {
            "precision": precision,
            "recall": recall,
            "fmeasure": fmeasure,
            "overlap_count": overlap_count,
            "reference_count": ref_count,
            "candidate_count": cand_count,
            "n": self.n,
            "score_type": self.score_type,
        }

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate ROUGE N-gram score"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"ROUGE-{self.n} {self.score_type}: {score:.4f}",
            metadata=details,
        )


class ROUGE3Grader(ROUGENGramGrader):
    """
    ROUGE-3 Grader

    Measures 3-gram overlap between candidate and reference texts.

    Example:
        >>> grader = ROUGE3Grader()
        >>> result = await grader.evaluate(
        ...     reference="the quick brown fox jumps over the lazy dog",
        ...     candidate="the quick brown fox jumps over the lazy cat"
        ... )
        >>> print(f"ROUGE-3: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "rouge3",
        score_type: str = "fmeasure",
        description: str = "ROUGE-3 metric (3-gram overlap)",
    ):
        super().__init__(name=name, n=3, score_type=score_type, description=description)


class ROUGE4Grader(ROUGENGramGrader):
    """
    ROUGE-4 Grader

    Measures 4-gram overlap between candidate and reference texts.

    Example:
        >>> grader = ROUGE4Grader()
        >>> result = await grader.evaluate(
        ...     reference="the quick brown fox jumps over the lazy dog today",
        ...     candidate="the quick brown fox jumps over the lazy dog yesterday"
        ... )
        >>> print(f"ROUGE-4: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "rouge4",
        score_type: str = "fmeasure",
        description: str = "ROUGE-4 metric (4-gram overlap)",
    ):
        super().__init__(name=name, n=4, score_type=score_type, description=description)


class ROUGE5Grader(ROUGENGramGrader):
    """
    ROUGE-5 Grader

    Measures 5-gram overlap between candidate and reference texts.

    Example:
        >>> grader = ROUGE5Grader()
        >>> result = await grader.evaluate(
        ...     reference="the quick brown fox jumps over the lazy dog every single day",
        ...     candidate="the quick brown fox jumps over the lazy dog every single time"
        ... )
        >>> print(f"ROUGE-5: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "rouge5",
        score_type: str = "fmeasure",
        description: str = "ROUGE-5 metric (5-gram overlap)",
    ):
        super().__init__(name=name, n=5, score_type=score_type, description=description)


__all__ = [
    "ROUGENGramGrader",
    "ROUGE3Grader",
    "ROUGE4Grader",
    "ROUGE5Grader",
]
