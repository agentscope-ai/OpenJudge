# -*- coding: utf-8 -*-
"""Crossref verification: DOI lookup and title search."""

import urllib.parse
from typing import List, Optional, Tuple

import httpx

from cookbooks.ref_hallucination_arena.schema import (
    MatchDetail,
    Reference,
    VerificationResult,
    VerificationStatus,
)
from cookbooks.ref_hallucination_arena.verifiers.base_verifier import BaseVerifier


class CrossrefVerifier(BaseVerifier):
    """Verify references via Crossref API (DOI + title search)."""

    API_URL = "https://api.crossref.org/works"

    def __init__(self, mailto: Optional[str] = None, timeout: float = 30.0):
        self.mailto = mailto
        headers = {"User-Agent": f"RefArena/1.0 (mailto:{mailto})" if mailto else "RefArena/1.0"}
        self.client = httpx.Client(timeout=timeout, headers=headers)

    @property
    def source_name(self) -> str:
        return "crossref"

    def verify(self, ref: Reference) -> VerificationResult:
        """Verify via DOI first, then title search."""
        try:
            if ref.doi:
                result = self._verify_by_doi(ref)
                if result.status == VerificationStatus.VERIFIED:
                    return result

            return self._verify_by_title(ref)
        except httpx.RequestError as e:
            return self._error(ref, f"Crossref network error: {e}")
        except Exception as e:
            return self._error(ref, f"Crossref error: {e}")

    def _verify_by_doi(self, ref: Reference) -> VerificationResult:
        """Verify by DOI lookup with strict title/author/year validation.

        DOI existence alone is NOT sufficient.  The actual paper behind the
        DOI must match the reference's title, every provided author name,
        and the publication year.
        """
        url = f"{self.API_URL}/{urllib.parse.quote(ref.doi, safe='')}"
        resp = self.client.get(url)
        if resp.status_code != 200:
            return self._not_found(ref, "DOI not found in Crossref")

        data = resp.json().get("message", {})
        detail, author_names = self._extract_match_with_authors(ref, data)
        matched_doi = data.get("DOI", "")

        # Strict validation: title + year + all provided authors
        if self.strict_match_with_author_list(ref, detail, author_names, matched_doi):
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                confidence=1.0,
                message=self.format_match_message(detail, "DOI verified"),
                source=self.source_name,
                match_detail=detail,
                match_data=data,
            )

        # DOI exists but content doesn't match -> hallucination
        score = self.compute_overall_score(detail)
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.NOT_FOUND,
            confidence=score,
            message=self.format_match_message(detail, "DOI mismatch"),
            match_detail=detail,
        )

    def _verify_by_title(self, ref: Reference) -> VerificationResult:
        """Verify by title search with strict validation."""
        params = {"query.title": ref.title, "rows": 5}
        if self.mailto:
            params["mailto"] = self.mailto
        resp = self.client.get(self.API_URL, params=params)

        if resp.status_code != 200:
            return self._not_found(ref, "Crossref API error")

        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return self._not_found(ref, "Not found in Crossref")

        best_verified: Optional[Tuple[dict, MatchDetail, float]] = None
        best_partial: Optional[Tuple[dict, MatchDetail, float]] = None

        for item in items:
            detail, author_names = self._extract_match_with_authors(ref, item)
            matched_doi = item.get("DOI", "")
            score = self.compute_overall_score(detail)

            # Try strict match first
            if self.strict_match_with_author_list(ref, detail, author_names, matched_doi):
                if best_verified is None or score > best_verified[2]:
                    best_verified = (item, detail, score)
            elif score >= 0.4:
                if best_partial is None or score > best_partial[2]:
                    best_partial = (item, detail, score)

        if best_verified:
            item, detail, score = best_verified
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                confidence=score,
                message=self.format_match_message(detail, "Crossref"),
                source=self.source_name,
                match_detail=detail,
                match_data=item,
            )

        # Partial match → still hallucination, but preserve match_detail for stats
        if best_partial:
            item, detail, score = best_partial
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.NOT_FOUND,
                confidence=score,
                message=self.format_match_message(detail, "Partial match"),
                match_detail=detail,
            )

        return self._not_found(ref, "Not found in Crossref")

    def _extract_match_with_authors(self, ref: Reference, data: dict) -> Tuple[MatchDetail, List[str]]:
        """Extract match details and full author name list from Crossref response.

        Returns a tuple of (MatchDetail, list_of_author_full_names).
        """
        titles = data.get("title", [])
        matched_title = titles[0] if titles else ""
        title_sim = self.text_similarity(ref.title, matched_title)

        authors = data.get("author", [])
        author_names = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors]
        matched_authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            matched_authors_str += " et al."
        author_sim = self.author_similarity(ref.authors, author_names) if ref.authors else 1.0

        pub_date = data.get("published-print") or data.get("published-online") or data.get("created")
        matched_year = ""
        if pub_date and "date-parts" in pub_date:
            parts = pub_date["date-parts"][0]
            matched_year = str(parts[0]) if parts else ""
        year_match = ref.year == matched_year if ref.year and matched_year else True

        detail = MatchDetail(
            title_match=title_sim,
            author_match=author_sim,
            year_match=year_match,
            matched_title=matched_title,
            matched_authors=matched_authors_str,
            matched_year=matched_year,
        )

        return detail, author_names

    # Keep backward-compatible _extract_match
    def _extract_match(self, ref: Reference, data: dict) -> MatchDetail:
        """Extract match details from Crossref response."""
        detail, _ = self._extract_match_with_authors(ref, data)
        return detail

    def close(self) -> None:
        if self.client:
            self.client.close()
