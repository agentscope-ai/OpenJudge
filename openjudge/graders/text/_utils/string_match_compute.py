# -*- coding: utf-8 -*-
"""
String Match Computation Functions

Utility functions for various string matching algorithms.
Supports both English and CJK (Chinese/Japanese/Korean) text.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from openjudge.graders.text._utils.tokenization import word_tokenize


# pylint: disable=unused-argument
def compute_exact_match(
    reference: str,
    response: str,
    case_sensitive: bool = True,
    ignore_whitespace: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute exact match between reference and response

    Args:
        reference: Reference text
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching
        ignore_whitespace: Whether to ignore whitespace differences

    Returns:
        Tuple of (score, details)
    """
    ref_processed = reference
    cand_processed = response

    if not case_sensitive:
        ref_processed = ref_processed.lower()
        cand_processed = cand_processed.lower()

    if ignore_whitespace:
        ref_processed = "".join(ref_processed.split())
        cand_processed = "".join(cand_processed.split())

    matched = ref_processed == cand_processed

    details = {
        "matched": matched,
        "case_sensitive": case_sensitive,
        "ignore_whitespace": ignore_whitespace,
    }

    return 1.0 if matched else 0.0, details


# pylint: disable=unused-argument
def compute_prefix_match(
    reference: str,
    response: str,
    case_sensitive: bool = True,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if response starts with reference

    Args:
        reference: Reference text (prefix to match)
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    ref = reference
    cand = response

    if not case_sensitive:
        ref = ref.lower()
        cand = cand.lower()

    matched = cand.startswith(ref)

    details = {
        "matched": matched,
        "case_sensitive": case_sensitive,
    }

    return 1.0 if matched else 0.0, details


# pylint: disable=unused-argument
def compute_suffix_match(
    reference: str,
    response: str,
    case_sensitive: bool = True,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if response ends with reference

    Args:
        reference: Reference text (suffix to match)
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    ref = reference
    cand = response

    if not case_sensitive:
        ref = ref.lower()
        cand = cand.lower()

    matched = cand.endswith(ref)

    details = {
        "matched": matched,
        "case_sensitive": case_sensitive,
    }

    return 1.0 if matched else 0.0, details


def compute_regex_match(
    reference: str,
    response: str,
    pattern: str = "",
    case_sensitive: bool = True,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Match response using regular expression pattern

    Args:
        reference: Reference text (used as pattern if pattern not provided)
        response: Response text
        pattern: Regular expression pattern (overrides reference if provided)
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    # Use pattern if provided, otherwise use reference
    pattern_str = pattern if pattern else reference

    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        regex = re.compile(pattern_str, flags)
        match = regex.search(response)
        matched = match is not None

        details = {
            "matched": matched,
            "pattern": pattern_str,
            "case_sensitive": case_sensitive,
            "match_groups": match.groups() if match else None,
        }

        return 1.0 if matched else 0.0, details
    except re.error as e:
        return 0.0, {"error": f"Invalid regex pattern: {str(e)}"}


# pylint: disable=unused-argument
def compute_substring_match(
    reference: str,
    response: str,
    case_sensitive: bool = False,
    bidirectional: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if response contains reference (or vice versa if bidirectional)

    Args:
        reference: Reference text (substring to find)
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching
        bidirectional: Whether to check both directions

    Returns:
        Tuple of (score, details)
    """
    ref = reference
    cand = response

    if not case_sensitive:
        ref = ref.lower()
        cand = cand.lower()

    if bidirectional:
        matched = ref in cand or cand in ref
    else:
        matched = ref in cand

    details = {
        "matched": matched,
        "case_sensitive": case_sensitive,
        "bidirectional": bidirectional,
    }

    return 1.0 if matched else 0.0, details


# pylint: disable=unused-argument
def compute_contains_all(
    reference: str,
    response: str,
    substrings: Optional[List[str]] = None,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if response contains all specified substrings

    Args:
        reference: Reference text (used if substrings not provided)
        response: Response text
        substrings: List of substrings to check
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details) - score is proportion of contained substrings
    """
    # Use substrings if provided, otherwise use reference
    target_substrings = substrings if substrings else [reference]

    cand = response
    if not case_sensitive:
        cand = cand.lower()
        target_substrings = [s.lower() for s in target_substrings]

    contains = [substring in cand for substring in target_substrings]
    matched = all(contains)

    details = {
        "matched": matched,
        "num_substrings": len(target_substrings),
        "contains_per_substring": contains,
        "missing_substrings": [s for s, c in zip(target_substrings, contains) if not c],
        "case_sensitive": case_sensitive,
    }

    # Calculate score: proportion of contained substrings
    score = sum(contains) / len(contains) if contains else 0.0

    return score, details


# pylint: disable=unused-argument
def compute_contains_any(
    reference: str,
    response: str,
    substrings: Optional[List[str]] = None,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if response contains at least one of the specified substrings

    Args:
        reference: Reference text (used if substrings not provided)
        response: Response text
        substrings: List of substrings to check
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    # Use substrings if provided, otherwise use reference
    target_substrings = substrings if substrings else [reference]

    cand = response
    if not case_sensitive:
        cand = cand.lower()
        target_substrings = [s.lower() for s in target_substrings]

    contains = [substring in cand for substring in target_substrings]
    matched = any(contains)

    details = {
        "matched": matched,
        "num_substrings": len(target_substrings),
        "contains_per_substring": contains,
        "matched_substrings": [s for s, c in zip(target_substrings, contains) if c],
        "case_sensitive": case_sensitive,
    }

    return 1.0 if matched else 0.0, details


# pylint: disable=unused-argument
def compute_word_overlap(
    reference: str,
    response: str,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate proportion of word overlap between response and reference

    Uses jieba tokenization with punctuation/stop-word removal so that
    high-frequency function words do not inflate the overlap score.

    Args:
        reference: Reference text
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    ref = reference
    cand = response

    if not case_sensitive:
        ref = ref.lower()
        cand = cand.lower()

    ref_words = set(word_tokenize(ref, remove_punctuation=True, remove_stopwords=True))
    cand_words = set(word_tokenize(cand, remove_punctuation=True, remove_stopwords=True))

    if len(ref_words) == 0:
        score = 0.0
    else:
        overlap = cand_words & ref_words
        score = len(overlap) / len(ref_words)

    details = {
        "overlap_ratio": score,
        "case_sensitive": case_sensitive,
        "num_ref_words": len(ref_words),
        "num_cand_words": len(cand_words),
        "num_overlap_words": len(cand_words & ref_words) if ref_words else 0,
    }

    return score, details


# pylint: disable=unused-argument
def compute_char_overlap(
    reference: str,
    response: str,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate proportion of character overlap between response and reference

    Args:
        reference: Reference text
        response: Response text
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Tuple of (score, details)
    """
    ref = reference
    cand = response

    if not case_sensitive:
        ref = ref.lower()
        cand = cand.lower()

    ref_chars = set(ref)
    cand_chars = set(cand)

    if len(ref_chars) == 0:
        score = 0.0
    else:
        overlap = cand_chars & ref_chars
        score = len(overlap) / len(ref_chars)

    details = {
        "overlap_ratio": score,
        "case_sensitive": case_sensitive,
        "num_ref_chars": len(ref_chars),
        "num_cand_chars": len(cand_chars),
        "num_overlap_chars": len(cand_chars & ref_chars) if ref_chars else 0,
    }

    return score, details
