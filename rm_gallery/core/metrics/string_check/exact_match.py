"""
Exact Match Metrics

Exact matching metrics for checking if candidate text matches reference text exactly.
Restructured to work with Grader framework.
"""

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class ExactMatchGrader(Grader):
    """
    Exact Match Grader

    Checks if the candidate text exactly matches the reference text.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching
        ignore_whitespace: Whether to ignore whitespace differences

    Example:
        >>> grader = ExactMatchGrader(case_sensitive=False)
        >>> result = await grader.evaluate(
        ...     reference="Hello World",
        ...     candidate="hello world"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "exact_match",
        case_sensitive: bool = True,
        ignore_whitespace: bool = False,
        description: str = "Exact match between reference and candidate",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace

    def _preprocess(self, text: str) -> str:
        """Preprocess text"""
        if not self.case_sensitive:
            text = text.lower()
        if self.ignore_whitespace:
            text = "".join(text.split())
        return text

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """
        Compute exact match

        Returns:
            tuple[bool, dict]: (matched, details)
        """
        reference_processed = self._preprocess(reference)
        candidate_processed = self._preprocess(candidate)
        matched = candidate_processed == reference_processed

        details = {
            "matched": matched,
            "case_sensitive": self.case_sensitive,
            "ignore_whitespace": self.ignore_whitespace,
        }

        return matched, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """
        Evaluate exact match

        Args:
            reference: Reference text
            candidate: Candidate text

        Returns:
            GraderScore: Score is 1.0 (matched) or 0.0 (not matched)
        """
        matched, details = self._compute(reference, candidate)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Exact match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )


class PrefixMatchGrader(Grader):
    """
    Prefix Match Grader

    Checks if the candidate text starts with the reference text.
    Similar to OpenAI Evals' Match metric.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = PrefixMatchGrader()
        >>> result = await grader.evaluate(
        ...     reference="Hello",
        ...     candidate="Hello World"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "prefix_match",
        case_sensitive: bool = True,
        description: str = "Check if candidate starts with reference",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """Compute prefix match"""
        if not self.case_sensitive:
            reference = reference.lower()
            candidate = candidate.lower()

        matched = candidate.startswith(reference)
        details = {"matched": matched, "case_sensitive": self.case_sensitive}

        return matched, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate prefix match"""
        matched, details = self._compute(reference, candidate)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Prefix match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )


class SuffixMatchGrader(Grader):
    """
    Suffix Match Grader

    Checks if the candidate text ends with the reference text.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = SuffixMatchGrader()
        >>> result = await grader.evaluate(
        ...     reference="World",
        ...     candidate="Hello World"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "suffix_match",
        case_sensitive: bool = True,
        description: str = "Check if candidate ends with reference",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """Compute suffix match"""
        if not self.case_sensitive:
            reference = reference.lower()
            candidate = candidate.lower()

        matched = candidate.endswith(reference)
        details = {"matched": matched, "case_sensitive": self.case_sensitive}

        return matched, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate suffix match"""
        matched, details = self._compute(reference, candidate)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Suffix match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )


class RegexMatchGrader(Grader):
    """
    Regular Expression Match Grader

    Matches candidate text using regular expression patterns.

    Attributes:
        name: Grader name
        pattern: Regular expression pattern (if not provided, uses reference as pattern)
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = RegexMatchGrader(pattern=r"\\d{3}-\\d{4}")
        >>> result = await grader.evaluate(
        ...     reference="",  # pattern already specified at initialization
        ...     candidate="My phone is 123-4567"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "regex_match",
        pattern: str = "",
        case_sensitive: bool = True,
        description: str = "Regular expression pattern matching",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.pattern = pattern
        self.case_sensitive = case_sensitive

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """Compute regex match"""
        import re

        # If no preset pattern, use reference as pattern
        pattern = self.pattern if self.pattern else reference

        flags = 0 if self.case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
            match = regex.search(candidate)
            matched = match is not None

            details = {
                "matched": matched,
                "pattern": pattern,
                "case_sensitive": self.case_sensitive,
                "match_groups": match.groups() if match else None,
            }

            return matched, details
        except re.error as e:
            return False, {"error": f"Invalid regex pattern: {str(e)}"}

    async def evaluate(
        self, reference: str = "", candidate: str = "", **kwargs
    ) -> GraderScore:
        """Evaluate regex match"""
        matched, details = self._compute(reference, candidate)

        if "error" in details:
            return GraderScore(score=0.0, reason=details["error"], metadata=details)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Regex match: {'matched' if matched else 'not matched'}",
            metadata=details,
        )


__all__ = [
    "ExactMatchGrader",
    "PrefixMatchGrader",
    "SuffixMatchGrader",
    "RegexMatchGrader",
]
