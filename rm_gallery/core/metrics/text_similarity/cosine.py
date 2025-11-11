"""
Cosine Similarity Metric

Cosine similarity metrics based on TF-IDF or term frequency vectors.
Restructured to work with Grader framework.
"""

from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class CosineSimilarityGrader(Grader):
    """
    Cosine Similarity Grader (TF-IDF based)

    Converts text to TF-IDF vectors and calculates cosine similarity.

    Attributes:
        name: Grader name
        use_tfidf: Whether to use TF-IDF
        ngram_range: N-gram range
        max_features: Maximum number of features

    Example:
        >>> grader = CosineSimilarityGrader(use_tfidf=True)
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the mat"
        ... )
        >>> print(f"Cosine Similarity: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "cosine",
        use_tfidf: bool = True,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: Optional[int] = None,
        description: str = "Cosine similarity metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.use_tfidf = use_tfidf
        self.ngram_range = ngram_range
        self.max_features = max_features

    def _cosine_similarity_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return max(0.0, min(similarity, 1.0))

    def _cosine_tfidf(self, text1: str, text2: str) -> float:
        """TF-IDF based cosine similarity"""
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range, max_features=self.max_features
            )
            vectors = vectorizer.fit_transform([text1, text2])
            vec1 = vectors[0].toarray().flatten()
            vec2 = vectors[1].toarray().flatten()
        except Exception:
            return 0.0

        return self._cosine_similarity_vectors(vec1, vec2)

    def _cosine_simple(self, text1: str, text2: str) -> float:
        """Simple term frequency based cosine similarity"""
        words1 = text1.split()
        words2 = text2.split()

        counter1 = Counter(words1)
        counter2 = Counter(words2)

        all_words = set(counter1.keys()) | set(counter2.keys())
        if not all_words:
            return 0.0

        vec1 = np.array([counter1.get(word, 0) for word in all_words])
        vec2 = np.array([counter2.get(word, 0) for word in all_words])

        return self._cosine_similarity_vectors(vec1, vec2)

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute cosine similarity

        Returns:
            tuple[float, dict]: (similarity_score, details)
        """
        if self.use_tfidf:
            score = self._cosine_tfidf(candidate, reference)
        else:
            score = self._cosine_simple(candidate, reference)

        details = {
            "use_tfidf": self.use_tfidf,
            "ngram_range": self.ngram_range,
        }

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate cosine similarity"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Cosine similarity: {score:.4f}",
            metadata=details,
        )


class JaccardSimilarityGrader(Grader):
    """
    Jaccard Similarity Grader

    Calculates Jaccard similarity of word sets between two texts.

    Attributes:
        name: Grader name
        use_ngrams: Whether to use n-grams instead of words
        n: Size of n-grams (when use_ngrams=True)

    Example:
        >>> grader = JaccardSimilarityGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the mat"
        ... )
        >>> print(f"Jaccard Similarity: {result.score:.4f}")
    """

    def __init__(
        self,
        name: str = "jaccard",
        use_ngrams: bool = False,
        n: int = 2,
        description: str = "Jaccard similarity metric",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.use_ngrams = use_ngrams
        self.n = n

    def _ngram_tokenize(self, text: str, n: int) -> list:
        """Simple n-gram tokenization"""
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i : i + n]))
        return ngrams

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute Jaccard similarity"""
        if self.use_ngrams:
            tokens1 = set(self._ngram_tokenize(candidate, n=self.n))
            tokens2 = set(self._ngram_tokenize(reference, n=self.n))
        else:
            tokens1 = set(candidate.split())
            tokens2 = set(reference.split())

        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0, {"use_ngrams": self.use_ngrams}

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        if len(union) == 0:
            return 0.0, {"use_ngrams": self.use_ngrams}

        score = len(intersection) / len(union)

        details = {
            "use_ngrams": self.use_ngrams,
            "n": self.n if self.use_ngrams else None,
        }

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate Jaccard similarity"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Jaccard similarity: {score:.4f}",
            metadata=details,
        )


__all__ = [
    "CosineSimilarityGrader",
    "JaccardSimilarityGrader",
]
