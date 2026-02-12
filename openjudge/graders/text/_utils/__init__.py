# -*- coding: utf-8 -*-
"""
Grader Utilities

Evaluation grader utility functions, including text normalization, tokenization, and core computations.
"""

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
from openjudge.graders.text._utils.normalization import (
    normalize_text,
    normalize_text_advanced,
)
from openjudge.graders.text._utils.string_match_compute import (
    compute_char_overlap,
    compute_contains_all,
    compute_contains_any,
    compute_exact_match,
    compute_prefix_match,
    compute_regex_match,
    compute_substring_match,
    compute_suffix_match,
    compute_word_overlap,
)
from openjudge.graders.text._utils.tokenization import (
    simple_tokenize,
    smart_tokenize,
    word_tokenize,
)

__all__ = [
    "normalize_text",
    "normalize_text_advanced",
    "smart_tokenize",
    "simple_tokenize",
    "word_tokenize",
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
    "compute_exact_match",
    "compute_prefix_match",
    "compute_suffix_match",
    "compute_regex_match",
    "compute_substring_match",
    "compute_contains_all",
    "compute_contains_any",
    "compute_word_overlap",
    "compute_char_overlap",
]
