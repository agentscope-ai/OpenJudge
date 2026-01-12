# -*- coding: utf-8 -*-
"""
String Match Grader

A unified grader for string matching evaluation supporting multiple algorithms:
- Exact Match, Prefix Match, Suffix Match
- Regex Match
- Substring Match, Contains All, Contains Any
- Word Overlap, Character Overlap
"""

from typing import Any, Dict

from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
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

# Algorithm to compute function mapping
COMPUTE_FUNCTIONS: Dict[str, Any] = {
    "exact_match": compute_exact_match,
    "prefix_match": compute_prefix_match,
    "suffix_match": compute_suffix_match,
    "regex_match": compute_regex_match,
    "substring_match": compute_substring_match,
    "contains_all": compute_contains_all,
    "contains_any": compute_contains_any,
    "word_overlap": compute_word_overlap,
    "char_overlap": compute_char_overlap,
}

# Default parameters for each algorithm
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "exact_match": {"case_sensitive": True, "ignore_whitespace": False},
    "prefix_match": {"case_sensitive": True},
    "suffix_match": {"case_sensitive": True},
    "regex_match": {"pattern": "", "case_sensitive": True},
    "substring_match": {"case_sensitive": False, "bidirectional": False},
    "contains_all": {"substrings": None, "case_sensitive": False},
    "contains_any": {"substrings": None, "case_sensitive": False},
    "word_overlap": {"case_sensitive": False},
    "char_overlap": {"case_sensitive": False},
}


class StringMatchGrader(BaseGrader):
    """
    Unified String Match Grader

    A general-purpose grader for evaluating string matching using various algorithms.
    The specific algorithm is chosen at evaluation time, not at initialization.

    Supported Algorithms:
        - exact_match: Exact string match (with case and whitespace options)
        - prefix_match: Check if response starts with reference_response
        - suffix_match: Check if response ends with reference_response
        - regex_match: Regular expression pattern matching
        - substring_match: Check if response contains reference_response
        - contains_all: Check if response contains all specified substrings
        - contains_any: Check if response contains any of the specified substrings
        - word_overlap: Calculate word overlap ratio
        - char_overlap: Calculate character overlap ratio

    Example:
        >>> grader = StringMatchGrader(case_sensitive=False)
        >>>
        >>> # Use exact match algorithm
        >>> result = await grader.aevaluate(
        ...     reference_response="Hello World",
        ...     response="hello world",
        ...     algorithm="exact_match"
        ... )
        >>>
        >>> # Use substring match algorithm with override
        >>> result = await grader.aevaluate(
        ...     reference_response="cat",
        ...     response="The cat sat on the mat",
        ...     algorithm="substring_match",
        ...     case_sensitive=True  # override init setting
        ... )
        >>>
        >>> # Use regex match algorithm
        >>> result = await grader.aevaluate(
        ...     reference_response=r"\\d{3}-\\d{4}",
        ...     response="My phone is 123-4567",
        ...     algorithm="regex_match"
        ... )
        >>>
        >>> # Use contains all algorithm
        >>> result = await grader.aevaluate(
        ...     reference_response="",
        ...     response="The cat sat on the mat",
        ...     algorithm="contains_all",
        ...     substrings=["cat", "mat"]
        ... )
    """

    def __init__(
        self,
        name: str = "string_match",
        description: str = "Unified string matching grader",
        case_sensitive: bool = False,
        ignore_whitespace: bool = False,
        algorithm: str = "exact_match",
    ):
        """
        Initialize string match grader

        Args:
            name: Grader name
            description: Grader description
            case_sensitive: Default case sensitivity for matching algorithms
            ignore_whitespace: Default whitespace handling for exact match
            algorithm: Algorithm to use (exact_match, substring_match, etc.)
        """
        super().__init__(
            name=name,
            mode=GraderMode.POINTWISE,
            description=description,
        )
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace
        self.algorithm = algorithm

        if self.algorithm not in COMPUTE_FUNCTIONS:
            raise ValueError(
                f"Unknown self.algorithm '{self.algorithm}'. "
                f"Supported algorithms: {', '.join(sorted(COMPUTE_FUNCTIONS.keys()))}",
            )

    async def aevaluate(
        self,
        reference_response: str = "",
        response: str = "",
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate string matching using specified algorithm

        Args:
            reference_response: Reference text
            response: Generated text to evaluate
            **kwargs: Algorithm-specific parameters that override init defaults
                     (e.g., case_sensitive, ignore_whitespace, pattern, substrings, etc.)

        Returns:
            GraderScore with matching score and details

        Raises:
            ValueError: If algorithm is not supported
        """
        # Get compute function
        compute_fn = COMPUTE_FUNCTIONS[self.algorithm]

        # Build params: default algorithm params -> init config -> kwargs override
        params = {**DEFAULT_PARAMS.get(self.algorithm, {})}

        # Apply init-level configuration if applicable to the algorithm
        if "case_sensitive" in params and "case_sensitive" not in kwargs:
            params["case_sensitive"] = self.case_sensitive
        if "ignore_whitespace" in params and "ignore_whitespace" not in kwargs:
            params["ignore_whitespace"] = self.ignore_whitespace

        # Override with kwargs
        params.update(kwargs)

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

    def _format_reason(
        self,
        algorithm: str,
        score: float,
        details: Dict[str, Any],
    ) -> str:
        """Format the reason string based on algorithm"""
        # Handle match-based algorithms
        if algorithm in ("exact_match", "prefix_match", "suffix_match", "regex_match"):
            matched = details.get("matched", False)
            matched_text = "matched" if matched else "not matched"
            return f"{algorithm.replace('_', ' ').title()}: {matched_text}"

        if algorithm in ("substring_match", "contains_any"):
            matched = details.get("matched", False)
            matched_text = "found" if matched else "not found"
            return f"{algorithm.replace('_', ' ').title()}: {matched_text}"

        # Handle contains_all separately due to complex logic
        if algorithm == "contains_all":
            matched = details.get("matched", False)
            num_substrings = details.get("num_substrings", 0)
            missing = details.get("missing_substrings", [])
            result = (
                f"Contains all {num_substrings} substrings"
                if matched
                else f"Missing {len(missing)} of {num_substrings} substrings"
            )
            return result

        # Handle overlap algorithms
        if algorithm in ("word_overlap", "char_overlap"):
            overlap_ratio = details.get("overlap_ratio", score)
            algo_type = "Word" if algorithm == "word_overlap" else "Character"
            return f"{algo_type} overlap ratio: {overlap_ratio:.2f}"

        # Default case
        return f"{algorithm.upper()} score: {score:.4f}"


__all__ = ["StringMatchGrader"]
