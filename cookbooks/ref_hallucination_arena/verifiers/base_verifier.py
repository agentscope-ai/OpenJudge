# -*- coding: utf-8 -*-
"""Base verifier interface for reference verification."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Set

from cookbooks.ref_hallucination_arena.schema import (
    MatchDetail,
    Reference,
    VerificationResult,
    VerificationStatus,
)


class BaseVerifier(ABC):
    """Abstract base class for reference verifiers.

    Each verifier implements verification against a specific data source
    (Crossref, PubMed, arXiv, DBLP, etc.).
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this verification source."""

    @abstractmethod
    def verify(self, ref: Reference) -> VerificationResult:
        """Verify a single reference against this source.

        Args:
            ref: Reference to verify.

        Returns:
            VerificationResult with status and details.
        """

    def close(self) -> None:
        """Clean up resources. Override if needed."""

    # ---- Shared utility methods ----

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML/XML tags from text, replacing with spaces."""
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize a title to a canonical form for comparison.

        Steps:
        1. Strip HTML/XML tags (replace with spaces).
        2. Replace typographic dashes (en-dash, em-dash) with ASCII hyphen.
        3. Remove all remaining punctuation except hyphens inside words.
        4. Lowercase.
        5. Collapse whitespace and strip.

        The result is a space-separated sequence of lowercase words where
        intra-word hyphens are preserved (e.g. "metal-organic").
        """
        t = BaseVerifier._strip_html(title)
        # Typographic dashes -> ASCII hyphen
        t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", t)
        t = t.lower()
        # Remove punctuation except hyphens that sit between word chars
        # First protect intra-word hyphens by replacing them temporarily
        t = re.sub(r"(\w)-(\w)", r"\1HYPHPLACEHOLDER\2", t)
        t = re.sub(r"[^\w\s]", " ", t)
        t = t.replace("HYPHPLACEHOLDER", "-")
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def text_similarity(t1: str, t2: str) -> float:
        """Calculate Jaccard similarity on word sets.

        HTML tags are stripped before comparison.
        This is kept for backward-compatible scoring / reporting;
        strict verification uses ``strict_title_check`` instead.
        """
        t1 = BaseVerifier._strip_html(t1)
        t2 = BaseVerifier._strip_html(t2)
        s1 = set(re.sub(r"[^\w\s]", "", t1.lower()).split())
        s2 = set(re.sub(r"[^\w\s]", "", t2.lower()).split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    @staticmethod
    def _strip_latex(name: str) -> str:
        r"""Strip LaTeX accent commands from a name string.

        Handles patterns like ``{\v{Z}}`` → ``Z``, ``\'{e}`` → ``e``,
        ``{\"o}`` → ``o``, etc.  Also removes remaining braces.
        """
        # \cmd{X} → X  (e.g. \v{Z} → Z, \'{e} → e)
        name = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", name)
        # \cmd X  (single-char shorthand, e.g. \"o → o)
        name = re.sub(r"\\[a-zA-Z]+\s*([a-zA-Z])", r"\1", name)
        # Remove remaining braces
        name = re.sub(r"[{}]", "", name)
        return name

    @staticmethod
    def _normalize_last_name(name: str) -> str:
        """Normalize a last name for comparison.

        Strips LaTeX accent commands, apostrophes, hyphens, lowercases,
        and applies Unicode NFKD decomposition to strip combining accents
        so that e.g. ``{\\v{Z}}ídek`` matches ``Žídek``,
        ``O'Keeffe`` matches ``OKeeffe``, and ``Lévy-Leduc`` matches
        ``levyleduc``.
        """
        import unicodedata

        name = BaseVerifier._strip_latex(name)
        name = name.lower().strip()
        name = re.sub(r"['\u2019\u2018`]", "", name)  # apostrophes
        name = re.sub(r"[-\u2010\u2011]", "", name)  # hyphens
        # NFKD decomposition: Ž → Z + combining caron, then strip combiners
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        return name

    @staticmethod
    def _is_valid_last_name(name: str) -> bool:
        """Check whether a string looks like a plausible author last name.

        Rejects empty strings, pure punctuation ("..."), single chars, and
        common placeholder tokens.
        """
        if not name:
            return False
        # Must contain at least 2 alphabetic characters
        alpha_chars = re.sub(r"[^a-z]", "", name.lower())
        if len(alpha_chars) < 2:
            return False
        # Reject ellipsis / placeholder patterns
        if re.match(r"^[.\u2026]+$", name):
            return False
        return True

    @staticmethod
    def _extract_bib_last_names(bib_authors: str) -> Set[str]:
        """Extract last-name set from a BibTeX author string.

        Handles "Last, First and Last, First ..." as well as
        "First Last and First Last ...".  Skips "others" / "et al." /
        invalid tokens like "...".
        """
        names: Set[str] = set()
        for part in re.split(r"\s+and\s+", bib_authors, flags=re.IGNORECASE):
            part = part.strip()
            if not part:
                continue
            if re.match(r"^(others|et\s+al\.?)$", part, re.IGNORECASE):
                continue
            if "," in part:
                last = part.split(",")[0].strip()
            else:
                words = part.split()
                last = words[-1] if words else ""
            last = BaseVerifier._normalize_last_name(last)
            if last and BaseVerifier._is_valid_last_name(last):
                names.add(last)
        return names

    @staticmethod
    def _extract_matched_last_names(matched_authors: List[str]) -> Set[str]:
        """Extract last-name set from a list of full-name strings.

        Handles two common formats:
          - "First Middle Last"  → last = Last  (Crossref / arXiv / DBLP)
          - "Last AB"            → last = Last  (PubMed abbreviated style)

        The PubMed style is detected when the final token is 1-4 uppercase
        letters (author initials like ``SL``, ``JR``, ``JWMC``).
        """
        names: Set[str] = set()
        for name in matched_authors:
            words = name.strip().split()
            if not words:
                continue
            # Detect PubMed "LastName Initials" format:
            # last token is short (≤4 chars), all uppercase, no lowercase
            last_word = words[-1]
            if len(words) >= 2 and len(last_word) <= 4 and last_word.isalpha() and last_word == last_word.upper():
                # PubMed style: take everything before the initials as last name
                # Handle multi-word last names like "De Vos" → join all but last
                last = " ".join(words[:-1])
            else:
                last = last_word
            last = BaseVerifier._normalize_last_name(last)
            if last and BaseVerifier._is_valid_last_name(last):
                names.add(last)
        return names

    @staticmethod
    def author_similarity(bib_authors: str, matched_authors: List[str]) -> float:
        """Calculate Jaccard similarity on author last names."""
        if not bib_authors or not matched_authors:
            return 0.0

        bib_names = BaseVerifier._extract_bib_last_names(bib_authors)
        matched_names = BaseVerifier._extract_matched_last_names(matched_authors)

        if not bib_names or not matched_names:
            return 0.0

        intersection = len(bib_names & matched_names)
        union = len(bib_names | matched_names)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def strict_author_check(bib_authors: str, matched_authors: List[str]) -> bool:
        """Strict author verification.

        Every author last-name provided by the model (from the BibTeX
        ``author`` field) must appear in the actual author list returned
        by the verification source.  The model may omit authors (via
        "et al." / "and others"), but every name it *does* provide must
        be present.

        Returns True when all provided names are found, False otherwise.
        """
        if not bib_authors:
            return True

        bib_names = BaseVerifier._extract_bib_last_names(bib_authors)
        if not bib_names:
            return True

        if not matched_authors:
            return False

        matched_names = BaseVerifier._extract_matched_last_names(matched_authors)
        if not matched_names:
            return False

        # Every bib name must exist in matched names
        return bib_names.issubset(matched_names)

    @staticmethod
    def strict_title_check(ref_title: str, matched_title: str) -> bool:
        """Strict title verification — normalized exact match.

        Both titles are normalized (strip HTML, lowercase, remove
        punctuation, collapse whitespace) and then compared as ordered
        word sequences.  This ensures that the titles are genuinely the
        same paper rather than merely sharing overlapping keywords.

        Returns True only when the normalized word sequences are
        identical.
        """
        return BaseVerifier._normalize_title(ref_title) == BaseVerifier._normalize_title(matched_title)

    @staticmethod
    def strict_year_check(ref_year: Optional[str], matched_year: str) -> bool:
        """Strict year verification.  Years must be identical."""
        if not ref_year or not matched_year:
            return True
        return ref_year.strip() == matched_year.strip()

    @staticmethod
    def strict_match_with_author_list(
        ref: Reference,
        detail: MatchDetail,
        matched_author_names: List[str],
        matched_doi: str = "",
    ) -> bool:
        """Combined strict check using the full author name list.

        Also populates the per-field exact-match flags on *detail* so
        that downstream scoring can compute per-field accuracy rates.

        Returns True only when title, year, AND all provided author
        names pass strict verification.
        """
        detail.title_exact = BaseVerifier.strict_title_check(ref.title, detail.matched_title)
        detail.year_exact = BaseVerifier.strict_year_check(ref.year, detail.matched_year)
        detail.author_exact = BaseVerifier.strict_author_check(ref.authors or "", matched_author_names)
        # DOI exact: both present and identical after lowering / stripping
        if ref.doi and matched_doi:
            detail.doi_exact = ref.doi.strip().lower() == matched_doi.strip().lower()
        else:
            detail.doi_exact = False
        detail.matched_doi = matched_doi

        return detail.title_exact and detail.year_exact and detail.author_exact

    @staticmethod
    def compute_overall_score(detail: MatchDetail) -> float:
        """Compute overall match score: title 50%, author 30%, year 20%."""
        score = detail.title_match * 0.5 + detail.author_match * 0.3
        if detail.year_match:
            score += 0.2
        return score

    @staticmethod
    def format_match_message(detail: MatchDetail, source: str) -> str:
        """Format a human-readable match message."""
        parts = [f"T:{detail.title_match:.0%}", f"A:{detail.author_match:.0%}"]
        parts.append("Y:\u2713" if detail.year_match else "Y:\u2717")
        return f"{source} ({', '.join(parts)})"

    @staticmethod
    def _not_found(ref: Reference, message: str = "") -> VerificationResult:
        """Convenience: return a NOT_FOUND result."""
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.NOT_FOUND,
            message=message or "Not found",
        )

    @staticmethod
    def _suspect(ref: Reference, message: str = "", confidence: float = 0.0) -> VerificationResult:
        """Convenience: return a SUSPECT result."""
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.SUSPECT,
            message=message or "Suspect",
            confidence=confidence,
        )

    @staticmethod
    def _error(ref: Reference, message: str = "") -> VerificationResult:
        """Convenience: return an ERROR result."""
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.ERROR,
            message=message or "Verification error",
        )
