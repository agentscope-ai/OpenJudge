# -*- coding: utf-8 -*-
"""DBLP API verification for computer science references."""

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


class DblpVerifier(BaseVerifier):
    """Verify references via DBLP API.

    Effective for computer science publications (conferences and journals).
    """

    API_URL = "https://dblp.org/search/publ/api"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "RefArena/1.0"})

    @property
    def source_name(self) -> str:
        return "dblp"

    def verify(self, ref: Reference) -> VerificationResult:
        """Verify by DBLP title search with strict validation."""
        try:
            query = urllib.parse.quote(ref.title)
            url = f"{self.API_URL}?q={query}&format=json&h=5"
            resp = self.client.get(url)

            if resp.status_code != 200:
                return self._not_found(ref, "DBLP API error")

            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                return self._not_found(ref, "Not found on DBLP")

            best_verified: Optional[Tuple[dict, MatchDetail, float]] = None
            best_partial: Optional[Tuple[dict, MatchDetail, float]] = None

            for hit in hits:
                info = hit.get("info", {})
                detail, author_names = self._extract_match_with_authors(ref, info)
                score = self.compute_overall_score(detail)

                if self.strict_match_with_author_list(ref, detail, author_names):
                    if best_verified is None or score > best_verified[2]:
                        best_verified = (info, detail, score)
                elif score >= 0.4:
                    if best_partial is None or score > best_partial[2]:
                        best_partial = (info, detail, score)

            if best_verified:
                info, detail, score = best_verified
                return VerificationResult(
                    reference=ref,
                    status=VerificationStatus.VERIFIED,
                    confidence=score,
                    message=self.format_match_message(detail, "DBLP"),
                    source=self.source_name,
                    match_detail=detail,
                    match_data=info,
                )

            if best_partial:
                info, detail, score = best_partial
                return VerificationResult(
                    reference=ref,
                    status=VerificationStatus.NOT_FOUND,
                    confidence=score,
                    message=self.format_match_message(detail, "DBLP partial"),
                    match_detail=detail,
                )

            return self._not_found(ref, "Mismatch on DBLP")
        except httpx.RequestError as e:
            return self._error(ref, f"DBLP network error: {e}")
        except Exception as e:
            return self._error(ref, f"DBLP error: {e}")

    def _extract_match_with_authors(self, ref: Reference, info: dict) -> Tuple[MatchDetail, List[str]]:
        """Extract match details and author name list from DBLP response."""
        # Title
        matched_title = info.get("title", "").rstrip(".")
        title_sim = self.text_similarity(ref.title, matched_title)

        # Authors
        authors = info.get("authors", {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        author_names = [a.get("text", a) if isinstance(a, dict) else str(a) for a in authors]
        matched_authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            matched_authors_str += " et al."
        author_sim = self.author_similarity(ref.authors, author_names) if ref.authors else 1.0

        # Year
        matched_year = info.get("year", "")
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

    def _extract_match(self, ref: Reference, info: dict) -> MatchDetail:
        """Extract match details from DBLP response."""
        detail, _ = self._extract_match_with_authors(ref, info)
        return detail

    def close(self) -> None:
        if self.client:
            self.client.close()
