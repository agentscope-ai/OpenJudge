# -*- coding: utf-8 -*-
"""
Text Similarity Grader

A unified grader for text similarity evaluation supporting multiple algorithms:
- BLEU, Sentence BLEU, GLEU, ChrF, METEOR
- ROUGE (1, 2, L, N-gram variants)
- F1 Score, Token F1
- Fuzzy Match, Edit Distance
- Cosine Similarity, Jaccard Similarity
"""

from typing import Any, Dict

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from openjudge.graders.text._utils.compute import (
    compute_bleu_score,
    compute_chrf_score,
    compute_cosine_similarity,
    compute_edit_distance,
    compute_f1_score,
    compute_fuzzy_match,
    compute_gleu_score,
    compute_jaccard_similarity,
    compute_meteor_score,
    compute_rouge_ngram,
    compute_rouge_scores,
    compute_sentence_bleu,
)

# Algorithm to compute function mapping
COMPUTE_FUNCTIONS: Dict[str, Any] = {
    "bleu": compute_bleu_score,
    "sentence_bleu": compute_sentence_bleu,
    "gleu": compute_gleu_score,
    "chrf": compute_chrf_score,
    "meteor": compute_meteor_score,
    "rouge": compute_rouge_scores,
    "rouge1": compute_rouge_scores,
    "rouge2": compute_rouge_scores,
    "rougeL": compute_rouge_scores,
    "rouge_ngram": compute_rouge_ngram,
    "rouge3": compute_rouge_ngram,
    "rouge4": compute_rouge_ngram,
    "rouge5": compute_rouge_ngram,
    "f1_score": compute_f1_score,
    "token_f1": compute_f1_score,
    "fuzzy_match": compute_fuzzy_match,
    "edit_distance": compute_edit_distance,
    "cosine": compute_cosine_similarity,
    "jaccard": compute_jaccard_similarity,
}

# Default parameters for each algorithm
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "bleu": {"max_ngram_order": 4, "smooth_method": "exp", "effective_order": True},
    "sentence_bleu": {"weights": (0.25, 0.25, 0.25, 0.25), "smoothing_function": 1},
    "gleu": {"min_len": 1, "max_len": 4},
    "chrf": {"char_order": 6, "beta": 2.0},
    "meteor": {"alpha": 0.9, "beta": 3.0, "gamma": 0.5},
    "rouge": {
        "rouge_types": ["rouge1", "rouge2", "rougeL"],
        "use_stemmer": True,
        "score_key": "fmeasure",
    },
    "rouge1": {"rouge_types": ["rouge1"], "use_stemmer": True, "score_key": "fmeasure"},
    "rouge2": {"rouge_types": ["rouge2"], "use_stemmer": True, "score_key": "fmeasure"},
    "rougeL": {"rouge_types": ["rougeL"], "use_stemmer": True, "score_key": "fmeasure"},
    "rouge_ngram": {"n": 3, "score_type": "fmeasure"},
    "rouge3": {"n": 3, "score_type": "fmeasure"},
    "rouge4": {"n": 4, "score_type": "fmeasure"},
    "rouge5": {"n": 5, "score_type": "fmeasure"},
    "f1_score": {"normalize": True},
    "token_f1": {"normalize": True},
    "fuzzy_match": {"method": "ratio", "threshold": 0.8},
    "edit_distance": {"normalize_by_length": True},
    "cosine": {"use_tfidf": True, "ngram_range": (1, 2), "max_features": None},
    "jaccard": {"use_ngrams": False, "n": 2},
}


class SimilarityGrader(BaseGrader):
    """
    Unified Text Similarity Grader

    A general-purpose grader for evaluating text similarity using various algorithms.
    The specific algorithm is chosen at evaluation time, not at initialization.

    Supported Algorithms:
        - bleu: Standard BLEU score (sacrebleu)
        - sentence_bleu: Sentence-level BLEU (NLTK)
        - gleu: Google BLEU score
        - chrf: Character n-gram F-score
        - meteor: METEOR score
        - rouge/rouge1/rouge2/rougeL: ROUGE variants
        - rouge3/rouge4/rouge5/rouge_ngram: ROUGE N-gram variants
        - f1_score/token_f1: Token-based F1 score
        - fuzzy_match: Levenshtein-based fuzzy matching
        - edit_distance: Edit distance similarity
        - cosine: Cosine similarity (TF-IDF based)
        - jaccard: Jaccard similarity

    Example:
        >>> grader = SimilarityGrader(normalize=True)
        >>>
        >>> # Use BLEU algorithm
        >>> result = await grader.aevaluate(
        ...     reference_response="the cat is on the mat",
        ...     candidate="the cat is on the mat",
        ...     algorithm="bleu",
        ...     max_ngram_order=4
        ... )
        >>>
        >>> # Use ROUGE algorithm
        >>> result = await grader.aevaluate(
        ...     reference_response="the cat is on the mat",
        ...     candidate="the cat is on the mat",
        ...     algorithm="rouge1"
        ... )
        >>>
        >>> # Use F1 Score algorithm with override
        >>> result = await grader.aevaluate(
        ...     reference_response="hello world",
        ...     candidate="hello world",
        ...     algorithm="f1_score",
        ...     normalize=False  # override init setting
        ... )
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        use_stemmer: bool = True,
        algorithm: str = "bleu",
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize similarity grader

        Args:
            normalize: Default normalization behavior for applicable algorithms
            case_sensitive: Default case sensitivity for applicable algorithms
            use_stemmer: Default stemmer usage for ROUGE algorithms
            algorithm: Algorithm to use (bleu, rouge, f1_score, etc.)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="similarity",
            mode=GraderMode.POINTWISE,
            description="Unified text similarity grader",
            strategy=strategy,
        )
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.use_stemmer = use_stemmer
        self.algorithm = algorithm

        if self.algorithm not in COMPUTE_FUNCTIONS:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Supported algorithms: {', '.join(sorted(COMPUTE_FUNCTIONS.keys()))}",
            )
        self.kwargs = kwargs

    async def _aevaluate(
        self,
        reference_response: str,
        response: str,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate text similarity using specified algorithm

        Args:
            reference_response: Reference text
            response: Response text to evaluate
            **kwargs: Algorithm-specific parameters that override init defaults
                     (e.g., normalize, case_sensitive, use_stemmer, max_ngram_order, etc.)

        Returns:
            GraderScore with similarity score and details

        Raises:
            ValueError: If algorithm is not supported
        """

        # Get compute function
        compute_fn = COMPUTE_FUNCTIONS[self.algorithm]

        # Build params: default algorithm params -> init config -> kwargs override
        params = {**DEFAULT_PARAMS.get(self.algorithm, {})}

        # Apply init-level configuration if applicable to the algorithm
        if "normalize" in params and "normalize" not in kwargs:
            params["normalize"] = self.normalize
        if "case_sensitive" in params and "case_sensitive" not in kwargs:
            params["case_sensitive"] = self.case_sensitive
        if "use_stemmer" in params and "use_stemmer" not in kwargs:
            params["use_stemmer"] = self.use_stemmer

        # Override with kwargs
        params.update(kwargs)
        params.update(self.kwargs)

        # Special handling for METEOR NLTK data
        if self.algorithm == "meteor":
            self._ensure_nltk_data()

        # Call the compute function
        score, details = compute_fn(reference_response, response, **params)

        # Handle errors
        if "error" in details:
            error_msg = details.get("message", details["error"])
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=error_msg,
                metadata=details,
            )

        # Format reason based on algorithm
        reason = self._format_reason(self.algorithm, score, details)

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata={**details, "algorithm": self.algorithm},
        )

    def _ensure_nltk_data(self) -> None:
        """Ensure NLTK data is downloaded for METEOR"""
        try:
            import nltk

            for package in ["wordnet", "punkt", "omw-1.4"]:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
        except ImportError:
            pass

    def _format_reason(
        self,
        algorithm: str,
        score: float,
        details: Dict[str, Any],
    ) -> str:
        """Format the reason string based on algorithm"""
        if algorithm in ("f1_score", "token_f1"):
            precision = details.get("precision", 0)
            recall = details.get("recall", 0)
            return f"F1 score: {score:.4f} (P={precision:.3f}, R={recall:.3f})"
        elif algorithm == "fuzzy_match":
            matched = details.get("matched", False)
            matched_text = "matched" if matched else "not matched"
            method = details.get("method", "ratio")
            return f"Fuzzy match ({method}): {score:.4f} ({matched_text})"
        elif algorithm == "edit_distance":
            distance = details.get("raw_distance", 0)
            return f"Edit distance similarity: {score:.4f} (distance={distance})"
        elif algorithm in ("rouge_ngram", "rouge3", "rouge4", "rouge5"):
            n = details.get("n", 3)
            score_type = details.get("score_type", "fmeasure")
            return f"ROUGE-{n} {score_type}: {score:.4f}"
        elif algorithm == "sentence_bleu":
            return f"Sentence BLEU: {score:.4f}"
        else:
            return f"{algorithm.upper()} score: {score:.4f}"


__all__ = ["SimilarityGrader"]
