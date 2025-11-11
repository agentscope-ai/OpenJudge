"""
GLEU Metric

GLEU (Google-BLEU) is a variant of BLEU proposed by Google.
Optimized for sentence-level evaluation, particularly suitable for grammatical error correction tasks.
Restructured to work with Grader framework.
"""

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class GLEUGrader(Grader):
    """
    GLEU Grader

    GLEU is a sentence-level variant of BLEU with the following improvements:
    1. Better suited for sentence-level evaluation
    2. More friendly to short sentences
    3. Takes recall into account

    Attributes:
        name: Grader name
        min_len: Minimum n-gram length
        max_len: Maximum n-gram length

    Example:
        >>> grader = GLEUGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> print(f"GLEU: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "gleu",
        min_len: int = 1,
        max_len: int = 4,
        description: str = "GLEU metric for sentence-level evaluation",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.min_len = min_len
        self.max_len = max_len

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute GLEU score

        Returns:
            tuple[float, dict]: (score, details)
        """
        try:
            from nltk.translate.gleu_score import sentence_gleu
        except ImportError:
            return 0.0, {
                "error": "NLTK not installed",
                "message": "Please install: pip install nltk",
            }

        # Tokenization
        candidate_tokens = candidate.split()
        reference_tokens = [reference.split()]

        try:
            score = sentence_gleu(
                reference_tokens,
                candidate_tokens,
                min_len=self.min_len,
                max_len=self.max_len,
            )

            details = {
                "min_len": self.min_len,
                "max_len": self.max_len,
                "num_references": len(reference_tokens),
                "candidate_length": len(candidate_tokens),
            }

            return score, details
        except Exception as e:
            return 0.0, {"error": str(e)}

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate GLEU score"""
        score, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(
                score=0.0,
                reason=details.get("message", details["error"]),
                metadata=details,
            )

        return GraderScore(
            score=score,
            reason=f"GLEU score: {score:.4f}",
            metadata=details,
        )


class ChrFGrader(Grader):
    """
    ChrF Grader

    Character n-gram F-score, an F-score based on character-level n-grams.
    Particularly suitable for morphologically rich languages and low-resource languages.

    Attributes:
        name: Grader name
        n: Character n-gram size
        beta: Beta parameter for F-score (beta=1 for F1, beta=2 for F2, etc.)

    Example:
        >>> grader = ChrFGrader(n=6, beta=2)
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> print(f"ChrF: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "chrf",
        n: int = 6,
        beta: float = 2.0,
        description: str = "Character n-gram F-score metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.n = n
        self.beta = beta

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute ChrF score"""
        try:
            from sacrebleu.metrics import CHRF
        except ImportError:
            return 0.0, {
                "error": "sacrebleu not installed",
                "message": "Please install: pip install sacrebleu",
            }

        # sacrebleu ChrF requires reference text format
        refs = [[reference]]

        try:
            chrf = CHRF(char_order=self.n, beta=self.beta)
            result = chrf.corpus_score([candidate], refs)

            # sacrebleu returns scores in 0-100 range
            normalized_score = result.score / 100.0

            details = {
                "char_order": self.n,
                "beta": self.beta,
                "raw_score": result.score,
            }

            return normalized_score, details
        except Exception as e:
            return 0.0, {"error": str(e)}

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate ChrF score"""
        score, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(
                score=0.0,
                reason=details.get("message", details["error"]),
                metadata=details,
            )

        return GraderScore(
            score=score,
            reason=f"ChrF score: {score:.4f}",
            metadata=details,
        )


__all__ = [
    "GLEUGrader",
    "ChrFGrader",
]
