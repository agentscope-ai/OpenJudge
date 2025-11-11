"""
ROUGE Metric

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric,
primarily used for automatic summarization evaluation.
Restructured to work with Grader framework.
"""

from typing import List

from rouge_score import rouge_scorer

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class ROUGEGrader(Grader):
    """
    ROUGE Grader

    Supports:
    - ROUGE-1: Word overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-Lsum: Sentence-level longest common subsequence

    Attributes:
        name: Grader name
        rouge_types: List of ROUGE types
        use_stemmer: Whether to use stemming
        score_key: Which score to use (precision/recall/fmeasure)

    Example:
        >>> grader = ROUGEGrader(rouge_types=["rouge1", "rouge2", "rougeL"])
        >>> result = await grader.evaluate(
        ...     reference="the cat is on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> print(f"ROUGE: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "rouge",
        rouge_types: List[str] = None,
        use_stemmer: bool = True,
        score_key: str = "fmeasure",
        description: str = "ROUGE metric for summarization evaluation",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.use_stemmer = use_stemmer
        self.score_key = score_key
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=use_stemmer
        )

    def _get_score_value(self, score_obj) -> float:
        """Extract score value from Score object"""
        if self.score_key == "precision":
            return score_obj.precision
        elif self.score_key == "recall":
            return score_obj.recall
        else:  # fmeasure (default)
            return score_obj.fmeasure

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute ROUGE score

        Returns:
            tuple[float, dict]: (average_score, details)
        """
        scores = self.scorer.score(reference, candidate)

        aggregated = {
            rouge_type: self._get_score_value(scores[rouge_type])
            for rouge_type in self.rouge_types
        }

        # Compute overall score (average of all ROUGE types)
        avg_score = sum(aggregated.values()) / len(aggregated)

        details = {
            **aggregated,
            "rouge_types": self.rouge_types,
            "use_stemmer": self.use_stemmer,
            "score_key": self.score_key,
        }

        return avg_score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate ROUGE score"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"ROUGE score: {score:.4f}",
            metadata=details,
        )


class ROUGE1Grader(ROUGEGrader):
    """ROUGE-1 Grader (word-level overlap)"""

    def __init__(
        self,
        name: str = "rouge1",
        use_stemmer: bool = True,
        score_key: str = "fmeasure",
        description: str = "ROUGE-1 metric (word overlap)",
    ):
        super().__init__(
            name=name,
            rouge_types=["rouge1"],
            use_stemmer=use_stemmer,
            score_key=score_key,
            description=description,
        )


class ROUGE2Grader(ROUGEGrader):
    """ROUGE-2 Grader (bigram overlap)"""

    def __init__(
        self,
        name: str = "rouge2",
        use_stemmer: bool = True,
        score_key: str = "fmeasure",
        description: str = "ROUGE-2 metric (bigram overlap)",
    ):
        super().__init__(
            name=name,
            rouge_types=["rouge2"],
            use_stemmer=use_stemmer,
            score_key=score_key,
            description=description,
        )


class ROUGELGrader(ROUGEGrader):
    """ROUGE-L Grader (longest common subsequence)"""

    def __init__(
        self,
        name: str = "rougeL",
        use_stemmer: bool = True,
        score_key: str = "fmeasure",
        description: str = "ROUGE-L metric (longest common subsequence)",
    ):
        super().__init__(
            name=name,
            rouge_types=["rougeL"],
            use_stemmer=use_stemmer,
            score_key=score_key,
            description=description,
        )


__all__ = [
    "ROUGEGrader",
    "ROUGE1Grader",
    "ROUGE2Grader",
    "ROUGELGrader",
]
