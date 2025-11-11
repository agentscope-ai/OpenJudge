"""
Substring Match Metrics

Substring matching metrics for checking text containment relationships.
Restructured to work with Grader framework.
"""

from typing import List

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class SubstringMatchGrader(Grader):
    """
    Substring Match Grader

    Checks if the candidate text contains the reference text (or reference contains candidate).
    Similar to OpenAI Evals' Includes metric.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching
        bidirectional: Whether to check bidirectionally

    Example:
        >>> grader = SubstringMatchGrader(case_sensitive=False)
        >>> result = await grader.evaluate(
        ...     reference="cat",
        ...     candidate="The cat sat on the mat"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "substring_match",
        case_sensitive: bool = False,
        bidirectional: bool = False,
        description: str = "Check if candidate contains reference",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive
        self.bidirectional = bidirectional

    def _check_substring(self, candidate: str, reference: str) -> bool:
        """Check substring relationship"""
        if self.bidirectional:
            return reference in candidate or candidate in reference
        else:
            return reference in candidate

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """Compute substring match"""
        if not self.case_sensitive:
            reference = reference.lower()
            candidate = candidate.lower()

        matched = self._check_substring(candidate, reference)

        details = {
            "matched": matched,
            "case_sensitive": self.case_sensitive,
            "bidirectional": self.bidirectional,
        }

        return matched, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate substring match"""
        matched, details = self._compute(reference, candidate)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Substring match: {'found' if matched else 'not found'}",
            metadata=details,
        )


class ContainsAllGrader(Grader):
    """
    Contains All Grader

    Checks if the candidate text contains all specified substrings.

    Attributes:
        name: Grader name
        substrings: List of substrings that must be contained
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = ContainsAllGrader(substrings=["cat", "mat"])
        >>> result = await grader.evaluate(
        ...     reference="",  # substrings already specified at initialization
        ...     candidate="The cat sat on the mat"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "contains_all",
        substrings: List[str] = None,
        case_sensitive: bool = False,
        description: str = "Check if candidate contains all specified substrings",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.substrings = substrings or []
        self.case_sensitive = case_sensitive

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """
        Compute contains all

        Returns:
            tuple[float, dict]: (score, details) - score is proportion of contained substrings
        """
        # Use substrings if preset, otherwise use reference
        substrings = self.substrings if self.substrings else [reference]

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()
            substrings = [s.lower() for s in substrings]

        # Check each substring
        contains = [substring in candidate for substring in substrings]
        matched = all(contains)

        details = {
            "matched": matched,
            "num_substrings": len(substrings),
            "contains_per_substring": contains,
            "missing_substrings": [s for s, c in zip(substrings, contains) if not c],
        }

        # Calculate score: proportion of contained substrings
        score = sum(contains) / len(contains) if contains else 0.0

        return score, details

    async def evaluate(
        self, reference: str = "", candidate: str = "", **kwargs
    ) -> GraderScore:
        """Evaluate contains all"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Contains all substrings: {details['matched']}",
            metadata=details,
        )


class ContainsAnyGrader(Grader):
    """
    Contains Any Grader

    Checks if the candidate text contains at least one of the specified substrings.

    Attributes:
        name: Grader name
        substrings: List of candidate substrings
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = ContainsAnyGrader(substrings=["cat", "dog"])
        >>> result = await grader.evaluate(
        ...     reference="",
        ...     candidate="The cat sat on the mat"
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        name: str = "contains_any",
        substrings: List[str] = None,
        case_sensitive: bool = False,
        description: str = "Check if candidate contains any of the specified substrings",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.substrings = substrings or []
        self.case_sensitive = case_sensitive

    def _compute(self, reference: str, candidate: str) -> tuple[bool, dict]:
        """Compute contains any"""
        # Use substrings if preset, otherwise use reference
        substrings = self.substrings if self.substrings else [reference]

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()
            substrings = [s.lower() for s in substrings]

        # Check each substring
        contains = [substring in candidate for substring in substrings]
        matched = any(contains)

        details = {
            "matched": matched,
            "num_substrings": len(substrings),
            "contains_per_substring": contains,
            "matched_substrings": [s for s, c in zip(substrings, contains) if c],
        }

        return matched, details

    async def evaluate(
        self, reference: str = "", candidate: str = "", **kwargs
    ) -> GraderScore:
        """Evaluate contains any"""
        matched, details = self._compute(reference, candidate)

        return GraderScore(
            score=1.0 if matched else 0.0,
            reason=f"Contains any substring: {matched}",
            metadata=details,
        )


class WordOverlapGrader(Grader):
    """
    Word Overlap Grader

    Calculates the proportion of word overlap between candidate and reference texts.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = WordOverlapGrader()
        >>> result = await grader.evaluate(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the rug"
        ... )
        >>> print(f"Overlap: {result.score:.2f}")
    """

    def __init__(
        self,
        name: str = "word_overlap",
        case_sensitive: bool = False,
        description: str = "Calculate word overlap ratio",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive

    def _compute_overlap(self, candidate: str, reference: str) -> float:
        """Calculate word overlap between two texts"""
        cand_words = set(candidate.split())
        ref_words = set(reference.split())

        if len(ref_words) == 0:
            return 0.0

        overlap = cand_words & ref_words
        return len(overlap) / len(ref_words)

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute word overlap"""
        if not self.case_sensitive:
            reference = reference.lower()
            candidate = candidate.lower()

        score = self._compute_overlap(candidate, reference)
        details = {"overlap_ratio": score, "case_sensitive": self.case_sensitive}

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate word overlap"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Word overlap ratio: {score:.2f}",
            metadata=details,
        )


class CharacterOverlapGrader(Grader):
    """
    Character Overlap Grader

    Calculates the proportion of character overlap between candidate and reference texts.

    Attributes:
        name: Grader name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> grader = CharacterOverlapGrader()
        >>> result = await grader.evaluate(
        ...     reference="hello",
        ...     candidate="helo"
        ... )
        >>> print(f"Overlap: {result.score:.2f}")
    """

    def __init__(
        self,
        name: str = "char_overlap",
        case_sensitive: bool = False,
        description: str = "Calculate character overlap ratio",
    ):
        super().__init__(
            name=name, grader_mode=GraderMode.POINTWISE, description=description
        )
        self.case_sensitive = case_sensitive

    def _compute_char_overlap(self, candidate: str, reference: str) -> float:
        """Calculate character overlap ratio"""
        cand_chars = set(candidate)
        ref_chars = set(reference)

        if len(ref_chars) == 0:
            return 0.0

        overlap = cand_chars & ref_chars
        return len(overlap) / len(ref_chars)

    def _compute(self, reference: str, candidate: str) -> tuple[float, dict]:
        """Compute character overlap"""
        if not self.case_sensitive:
            reference = reference.lower()
            candidate = candidate.lower()

        score = self._compute_char_overlap(candidate, reference)
        details = {"overlap_ratio": score, "case_sensitive": self.case_sensitive}

        return score, details

    async def evaluate(self, reference: str, candidate: str, **kwargs) -> GraderScore:
        """Evaluate character overlap"""
        score, details = self._compute(reference, candidate)

        return GraderScore(
            score=score,
            reason=f"Character overlap ratio: {score:.2f}",
            metadata=details,
        )


__all__ = [
    "SubstringMatchGrader",
    "ContainsAllGrader",
    "ContainsAnyGrader",
    "WordOverlapGrader",
    "CharacterOverlapGrader",
]
