# -*- coding: utf-8 -*-
"""
Core Computation Functions for Graders

This module contains core computation functions used by various graders.
All computation logic is centralized here to avoid code duplication.

All word-level tokenization is powered by **jieba** (via the tokenization
module), which handles English, Chinese, and mixed text uniformly.
"""

# pylint: disable=unused-argument

from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np

from openjudge.graders.text._utils.tokenization import smart_tokenize
from openjudge.graders.text._utils.tokenization import word_tokenize as _word_tokenize


def compute_bleu_score(
    reference: str,
    response: str,
    max_ngram_order: int = 4,
    smooth_method: str = "exp",
    effective_order: bool = True,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute BLEU score using sacrebleu

    Returns:
        Tuple[float, dict]: (normalized_score [0, 1], details)
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return 0.0, {
            "error": "sacrebleu not installed. Please install: pip install sacrebleu",
        }

    bleu = BLEU(
        max_ngram_order=max_ngram_order,
        smooth_method=smooth_method,
        effective_order=effective_order,
    )
    refs = [[reference]]

    try:
        result = bleu.corpus_score([response], refs)
        normalized_score = result.score / 100.0
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


def compute_sentence_bleu(
    reference: str,
    response: str,
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
    smoothing_function: int = 1,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute sentence-level BLEU score using NLTK.

    Tokenization is done via jieba (removes punctuation + stop-words)
    so the score reflects meaningful n-gram overlap for any language.

    Returns:
        Tuple[float, dict]: (score, details)
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return 0.0, {"error": "NLTK not installed. Please install: pip install nltk"}

    response_tokens = _word_tokenize(response, remove_punctuation=True, remove_stopwords=True)
    reference_tokens = [_word_tokenize(reference, remove_punctuation=True, remove_stopwords=True)]

    smoothing = SmoothingFunction()
    smooth_func = getattr(smoothing, f"method{smoothing_function}")

    try:
        score = sentence_bleu(
            reference_tokens,
            response_tokens,
            weights=weights,
            smoothing_function=smooth_func,
        )
        details = {
            "weights": weights,
            "smoothing_method": smoothing_function,
            "num_references": len(reference_tokens),
        }
        return score, details
    except Exception as e:
        return 0.0, {"error": str(e)}


def compute_gleu_score(
    reference: str,
    response: str,
    min_len: int = 1,
    max_len: int = 4,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute GLEU score using NLTK.

    Returns:
        Tuple[float, dict]: (score, details)
    """
    try:
        from nltk.translate.gleu_score import sentence_gleu
    except ImportError:
        return 0.0, {
            "error": "NLTK not installed",
            "message": "Please install: pip install nltk",
        }

    response_tokens = _word_tokenize(response, remove_punctuation=True, remove_stopwords=True)
    reference_tokens = [_word_tokenize(reference, remove_punctuation=True, remove_stopwords=True)]

    try:
        score = sentence_gleu(
            reference_tokens,
            response_tokens,
            min_len=min_len,
            max_len=max_len,
        )
        details = {
            "min_len": min_len,
            "max_len": max_len,
            "num_references": len(reference_tokens),
            "response_length": len(response_tokens),
        }
        return score, details
    except Exception as e:
        return 0.0, {"error": str(e)}


def compute_chrf_score(
    reference: str,
    response: str,
    char_order: int = 6,
    beta: float = 2.0,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute ChrF score using sacrebleu

    Returns:
        Tuple[float, dict]: (normalized_score [0, 1], details)
    """
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return 0.0, {
            "error": "sacrebleu not installed",
            "message": "Please install: pip install sacrebleu",
        }

    refs = [[reference]]

    try:
        chrf = CHRF(char_order=char_order, beta=beta)
        result = chrf.corpus_score([response], refs)
        normalized_score = result.score / 100.0

        details = {
            "char_order": char_order,
            "beta": beta,
            "raw_score": result.score,
        }
        return normalized_score, details
    except Exception as e:
        return 0.0, {"error": str(e)}


def compute_meteor_score(
    reference: str,
    response: str,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute METEOR score using NLTK.

    Note: NLTK METEOR's synonym/stemmer features are English-only;
    for CJK the score is based purely on token overlap and ordering.

    Returns:
        Tuple[float, dict]: (score, details)
    """
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        return 0.0, {
            "error": "NLTK not installed or missing dependencies",
            "message": "Please install: pip install nltk",
        }

    response_tokens = _word_tokenize(response, remove_punctuation=True)
    reference_tokens = _word_tokenize(reference, remove_punctuation=True)

    try:
        score = meteor_score(
            [reference_tokens],
            response_tokens,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        details = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
        return score, details
    except Exception as e:
        return 0.0, {"error": str(e)}


def compute_rouge_scores(
    reference: str,
    response: str,
    rouge_types: List[str],
    use_stemmer: bool = True,
    score_key: str = "fmeasure",
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute ROUGE scores using rouge_score library.

    Uses a jieba-based tokenizer so ROUGE works correctly for CJK text.
    English text also goes through jieba (which preserves English words).

    Returns:
        Tuple[float, dict]: (average_score, details)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0, {
            "error": "rouge_score not installed",
            "message": "Please install: pip install rouge-score",
        }

    _stemmer = None
    if use_stemmer:
        try:
            from nltk.stem.porter import PorterStemmer

            _stemmer = PorterStemmer()
        except ImportError:
            pass

    class _JiebaTokenizer:
        """Tokenizer adapter for rouge_score using jieba + optional stemming."""

        def tokenize(self, text):
            """Tokenize via jieba; apply Porter stemmer if enabled."""
            tokens = _word_tokenize(text, remove_punctuation=True, remove_stopwords=False)
            if _stemmer is not None:
                tokens = [_stemmer.stem(t) for t in tokens]
            return tokens

    try:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False, tokenizer=_JiebaTokenizer())
    except TypeError:
        # Older versions of rouge_score may not support tokenizer param
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)

    scores = scorer.score(reference, response)

    aggregated = {}
    for rouge_type in rouge_types:
        score_obj = scores[rouge_type]
        if score_key == "precision":
            aggregated[rouge_type] = score_obj.precision
        elif score_key == "recall":
            aggregated[rouge_type] = score_obj.recall
        else:  # fmeasure
            aggregated[rouge_type] = score_obj.fmeasure

    avg_score = sum(aggregated.values()) / len(aggregated)

    details = {
        **aggregated,
        "rouge_types": rouge_types,
        "use_stemmer": use_stemmer,
        "score_key": score_key,
    }

    return avg_score, details


def compute_rouge_ngram(
    reference: str,
    response: str,
    n: int,
    score_type: str = "fmeasure",
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute ROUGE N-gram score (custom implementation).

    Returns:
        Tuple[float, dict]: (score, details)
    """

    def _get_ngrams(text: str, n: int) -> List[tuple]:
        tokens = _word_tokenize(text, remove_punctuation=True, remove_stopwords=True)
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return ngrams

    ref_ngrams = _get_ngrams(reference, n)
    cand_ngrams = _get_ngrams(response, n)

    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)

    overlap = cand_counter & ref_counter
    overlap_count = sum(overlap.values())

    ref_count = sum(ref_counter.values())
    cand_count = sum(cand_counter.values())

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

    if score_type == "precision":
        score = precision
    elif score_type == "recall":
        score = recall
    else:
        score = fmeasure

    details = {
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure,
        "overlap_count": overlap_count,
        "reference_count": ref_count,
        "response_count": cand_count,
        "n": n,
        "score_type": score_type,
    }

    return score, details


def compute_f1_score(
    reference: str,
    response: str,
    normalize: bool = True,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute token-based F1 score.

    Uses jieba tokenization with punctuation/stop-word removal so that
    function words do not inflate the overlap.

    Returns:
        Tuple[float, dict]: (f1_score, details)
    """
    if normalize:
        response_norm = response.lower().strip()
        reference_norm = reference.lower().strip()
    else:
        response_norm = response
        reference_norm = reference

    response_tokens = _word_tokenize(response_norm, remove_punctuation=True, remove_stopwords=True)
    reference_tokens = _word_tokenize(reference_norm, remove_punctuation=True, remove_stopwords=True)

    if len(response_tokens) == 0 or len(reference_tokens) == 0:
        if len(response_tokens) == 0 and len(reference_tokens) == 0:
            return 1.0, {"precision": 1.0, "recall": 1.0}
        return 0.0, {"precision": 0.0, "recall": 0.0}

    response_counter = Counter(response_tokens)
    reference_counter = Counter(reference_tokens)
    common = response_counter & reference_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, {"precision": 0.0, "recall": 0.0}

    precision = 1.0 * num_same / len(response_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    details = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "common_tokens": num_same,
    }

    return f1, details


def compute_fuzzy_match(
    reference: str,
    response: str,
    method: str = "ratio",
    threshold: float = 0.8,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute fuzzy match score using Levenshtein distance

    Returns:
        Tuple[float, dict]: (score, details)
    """
    try:
        import Levenshtein
    except ImportError:
        return 0.0, {
            "error": "python-Levenshtein not installed. Please install: pip install python-Levenshtein",
        }

    if method == "ratio":
        score = Levenshtein.ratio(response, reference)
    elif method == "partial_ratio":
        score = _partial_ratio(response, reference)
    elif method == "token_sort_ratio":
        score = _token_sort_ratio(response, reference)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'ratio', 'partial_ratio', or 'token_sort_ratio'",
        )

    matched = score >= threshold

    details = {
        "method": method,
        "threshold": threshold,
        "matched": matched,
    }

    return score, details


def _partial_ratio(s1: str, s2: str) -> float:
    """Partial string matching"""
    import Levenshtein

    if len(s1) == 0 or len(s2) == 0:
        return 0.0 if s1 != s2 else 1.0

    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    m = len(shorter)
    max_ratio = 0.0

    for i in range(len(longer) - m + 1):
        ratio = Levenshtein.ratio(shorter, longer[i : i + m])
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def _token_sort_ratio(s1: str, s2: str) -> float:
    """Token order-independent fuzzy matching"""
    import Levenshtein

    tokens1 = sorted(smart_tokenize(s1))
    tokens2 = sorted(smart_tokenize(s2))
    return Levenshtein.ratio(" ".join(tokens1), " ".join(tokens2))


def compute_edit_distance(
    reference: str,
    response: str,
    normalize_by_length: bool = True,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute edit distance similarity

    Returns:
        Tuple[float, dict]: (normalized_score, details)
    """
    try:
        import Levenshtein
    except ImportError:
        return 0.0, {
            "error": "python-Levenshtein not installed. Please install: pip install python-Levenshtein",
        }

    raw_distance = Levenshtein.distance(response, reference)
    max_len = max(len(response), len(reference))

    if normalize_by_length and max_len > 0:
        normalized_score = 1.0 - (raw_distance / max_len)
    else:
        normalized_score = 1.0 / (1.0 + raw_distance)

    normalized_score = max(0.0, min(1.0, normalized_score))

    details = {
        "raw_distance": raw_distance,
        "max_length": max_len,
        "normalize_by_length": normalize_by_length,
    }

    return normalized_score, details


def compute_cosine_similarity(
    reference: str,
    response: str,
    use_tfidf: bool = True,
    ngram_range: tuple = (1, 2),
    max_features: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute cosine similarity

    Returns:
        Tuple[float, dict]: (similarity_score, details)
    """
    if use_tfidf:
        score = _cosine_tfidf(response, reference, ngram_range, max_features)
    else:
        score = _cosine_simple(response, reference)

    details = {
        "use_tfidf": use_tfidf,
        "ngram_range": ngram_range,
    }

    return score, details


def _cosine_similarity_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return max(0.0, min(similarity, 1.0))


def _jieba_tfidf_tokenizer(text: str) -> List[str]:
    """Custom tokenizer for TfidfVectorizer using jieba."""
    return _word_tokenize(text, remove_punctuation=True, remove_stopwords=True)


def _cosine_tfidf(
    text1: str,
    text2: str,
    ngram_range: tuple,
    max_features: Optional[int],
    **kwargs: Any,
) -> float:
    """TF-IDF based cosine similarity (jieba tokenizer)"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            tokenizer=_jieba_tfidf_tokenizer,
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=None,  # disable default pattern when using custom tokenizer
        )
        vectors = vectorizer.fit_transform([text1, text2])
        vec1 = vectors[0].toarray().flatten()
        vec2 = vectors[1].toarray().flatten()
    except Exception:
        return 0.0

    return _cosine_similarity_vectors(vec1, vec2)


def _cosine_simple(text1: str, text2: str) -> float:
    """Simple term frequency based cosine similarity"""
    words1 = _word_tokenize(text1, remove_punctuation=True, remove_stopwords=True)
    words2 = _word_tokenize(text2, remove_punctuation=True, remove_stopwords=True)

    counter1 = Counter(words1)
    counter2 = Counter(words2)

    all_words = set(counter1.keys()) | set(counter2.keys())
    if not all_words:
        return 0.0

    vec1 = np.array([counter1.get(word, 0) for word in all_words])
    vec2 = np.array([counter2.get(word, 0) for word in all_words])

    return _cosine_similarity_vectors(vec1, vec2)


def compute_jaccard_similarity(
    reference: str,
    response: str,
    use_ngrams: bool = False,
    n: int = 2,
    **kwargs: Any,
) -> Tuple[float, dict]:
    """
    Compute Jaccard similarity

    Returns:
        Tuple[float, dict]: (similarity_score, details)
    """
    if use_ngrams:
        tokens1 = set(_ngram_tokenize(response, n))
        tokens2 = set(_ngram_tokenize(reference, n))
    else:
        tokens1 = set(_word_tokenize(response, remove_punctuation=True, remove_stopwords=True))
        tokens2 = set(_word_tokenize(reference, remove_punctuation=True, remove_stopwords=True))

    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0, {"use_ngrams": use_ngrams}

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if len(union) == 0:
        return 0.0, {"use_ngrams": use_ngrams}

    score = len(intersection) / len(union)

    details = {
        "use_ngrams": use_ngrams,
        "n": n if use_ngrams else None,
    }

    return score, details


def _ngram_tokenize(text: str, n: int) -> list:
    """N-gram tokenization via jieba with stop-word removal."""
    words = _word_tokenize(text, remove_punctuation=True, remove_stopwords=True)
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i : i + n]))
    return ngrams


__all__ = [
    "compute_bleu_score",
    "compute_sentence_bleu",
    "compute_gleu_score",
    "compute_chrf_score",
    "compute_meteor_score",
    "compute_rouge_scores",
    "compute_rouge_ngram",
    "compute_f1_score",
    "compute_fuzzy_match",
    "compute_edit_distance",
    "compute_cosine_similarity",
    "compute_jaccard_similarity",
]
