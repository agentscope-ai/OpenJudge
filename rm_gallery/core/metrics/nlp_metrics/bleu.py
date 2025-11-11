"""
BLEU Metric

BLEU (Bilingual Evaluation Understudy) metric, the standard metric for machine translation evaluation.
Restructured to work with Grader framework.
"""

from sacrebleu.metrics import BLEU

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class BLEUGrader(Grader):
    """
    BLEU Grader

    Standard BLEU scoring using the sacrebleu library.

    Attributes:
        name: Grader name
        max_ngram_order: Maximum n-gram order (typically 4)
        smooth_method: Smoothing method (exp/floor/add-k/none)
        effective_order: Whether to use effective order

    Example:
        >>> grader = BLEUGrader(max_ngram_order=4)
        >>> result = await grader.evaluate(
        ...     reference="the cat is on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> print(f"BLEU: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "bleu",
        max_ngram_order: int = 4,
        smooth_method: str = "exp",
        effective_order: bool = True,
        description: str = "BLEU metric for translation evaluation",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.max_ngram_order = max_ngram_order
        self.smooth_method = smooth_method
        self.effective_order = effective_order
        self.bleu = BLEU(
            max_ngram_order=max_ngram_order,
            smooth_method=smooth_method,
            effective_order=effective_order,
        )

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute BLEU score

        Returns:
            tuple[float, dict]: (normalized_score [0, 1], details)
        """
        # sacrebleu requires reference text format: List[List[str]]
        refs = [[reference]]

        try:
            result = self.bleu.corpus_score([candidate], refs)

            # sacrebleu returns scores in 0-100 range
            normalized_score = result.score / 100.0
            # Clamp to [0, 1] range
            normalized_score = max(0.0, min(1.0, normalized_score))

            details = {
                "precisions": [p / 100.0 for p in result.precisions],
                "bp": result.bp,
                "sys_len": result.sys_len,
                "ref_len": result.ref_len,
                "ratio": result.sys_len / result.ref_len if result.ref_len > 0 else 0,
                "raw_score": result.score,
            }

            return normalized_score, details
        except Exception as e:
            return 0.0, {"error": str(e)}

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate BLEU score"""
        score, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(score=0.0, reason=details["error"], metadata=details)

        return GraderScore(
            score=score,
            reason=f"BLEU score: {score:.4f}",
            metadata=details,
        )


class SentenceBLEUGrader(Grader):
    """
    Sentence-level BLEU Grader

    Uses NLTK's sentence_bleu for single sentence evaluation.

    Attributes:
        name: Grader name
        weights: Weights for each n-gram order
        smoothing_function: Smoothing function type (1-7)

    Example:
        >>> grader = SentenceBLEUGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> print(f"Sentence BLEU: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "sentence_bleu",
        weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        smoothing_function: int = 1,
        description: str = "Sentence-level BLEU metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.weights = weights
        self.smoothing_function = smoothing_function

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute sentence-level BLEU score"""
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        except ImportError:
            return 0.0, {
                "error": "NLTK not installed. Please install: pip install nltk"
            }

        # Tokenize
        candidate_tokens = candidate.split()
        reference_tokens = [reference.split()]

        # Select smoothing function
        smoothing = SmoothingFunction()
        smooth_func = getattr(smoothing, f"method{self.smoothing_function}")

        try:
            score = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=self.weights,
                smoothing_function=smooth_func,
            )

            details = {
                "weights": self.weights,
                "smoothing_method": self.smoothing_function,
                "num_references": len(reference_tokens),
            }

            return score, details
        except Exception as e:
            return 0.0, {"error": str(e)}

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate sentence BLEU"""
        score, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(score=0.0, reason=details["error"], metadata=details)

        return GraderScore(
            score=score,
            reason=f"Sentence BLEU: {score:.4f}",
            metadata=details,
        )


__all__ = [
    "BLEUGrader",
    "SentenceBLEUGrader",
]
