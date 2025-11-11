"""
METEOR Metric

METEOR (Metric for Evaluation of Translation with Explicit ORdering),
a translation evaluation metric that comprehensively considers precision, recall,
morphological variations, and semantic information.
Restructured to work with Grader framework.
"""

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class METEORGrader(Grader):
    """
    METEOR Grader

    METEOR has the following improvements over BLEU:
    1. Considers both precision and recall
    2. Supports stemming and synonym matching
    3. Considers word order (via fragmentation penalty)

    Attributes:
        name: Grader name
        alpha: Precision weight parameter
        beta: Recall weight parameter
        gamma: Fragmentation penalty weight parameter

    Example:
        >>> grader = METEORGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="on the mat sat the cat"
        ... )
        >>> print(f"METEOR: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "meteor",
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        description: str = "METEOR metric for translation evaluation",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure NLTK data is downloaded"""
        try:
            import nltk

            for package in ["wordnet", "punkt", "omw-1.4"]:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
        except ImportError:
            pass

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute METEOR score

        Returns:
            tuple[float, dict]: (score, details)
        """
        try:
            from nltk.translate.meteor_score import meteor_score
        except ImportError:
            return 0.0, {
                "error": "NLTK not installed or missing dependencies",
                "message": "Please install: pip install nltk",
            }

        # Tokenization
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()

        try:
            score = meteor_score(
                [reference_tokens],
                candidate_tokens,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            details = {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            }

            return score, details
        except Exception as e:
            return 0.0, {"error": str(e)}

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate METEOR score"""
        score, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(
                score=0.0,
                reason=details.get("message", details["error"]),
                metadata=details,
            )

        return GraderScore(
            score=score,
            reason=f"METEOR score: {score:.4f}",
            metadata=details,
        )


__all__ = [
    "METEORGrader",
]
