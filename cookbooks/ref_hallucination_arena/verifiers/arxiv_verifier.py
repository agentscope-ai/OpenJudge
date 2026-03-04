# -*- coding: utf-8 -*-
"""arXiv API verification for preprint references."""

import re
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


class ArxivVerifier(BaseVerifier):
    """Verify references via arXiv API.

    Effective for preprints and CS/physics/math papers.
    """

    API_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "RefArena/1.0"})

    @property
    def source_name(self) -> str:
        return "arxiv"

    def verify(self, ref: Reference) -> VerificationResult:
        """Verify via arXiv ID or title search with strict validation."""
        try:
            if ref.arxiv_id:
                query = f"id:{ref.arxiv_id}"
            else:
                query = f"ti:{urllib.parse.quote(ref.title)}"

            url = f"{self.API_URL}?search_query={query}&max_results=5"
            resp = self.client.get(url)

            if resp.status_code != 200:
                return self._not_found(ref, "arXiv API error")

            entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
            if not entries:
                return self._not_found(ref, "Not found on arXiv")

            best_verified: Optional[Tuple[MatchDetail, float]] = None
            best_partial: Optional[Tuple[MatchDetail, float]] = None

            for entry in entries:
                detail, author_names = self._extract_match_with_authors(ref, entry)
                score = self.compute_overall_score(detail)

                if self.strict_match_with_author_list(ref, detail, author_names):
                    if best_verified is None or score > best_verified[1]:
                        best_verified = (detail, score)
                elif score >= 0.4:
                    if best_partial is None or score > best_partial[1]:
                        best_partial = (detail, score)

            if best_verified:
                detail, score = best_verified
                return VerificationResult(
                    reference=ref,
                    status=VerificationStatus.VERIFIED,
                    confidence=score,
                    message=self.format_match_message(detail, "arXiv"),
                    source=self.source_name,
                    match_detail=detail,
                )

            if best_partial:
                detail, score = best_partial
                return VerificationResult(
                    reference=ref,
                    status=VerificationStatus.NOT_FOUND,
                    confidence=score,
                    message=self.format_match_message(detail, "arXiv partial"),
                    match_detail=detail,
                )

            return self._not_found(ref, "Mismatch on arXiv")
        except httpx.RequestError as e:
            return self._error(ref, f"arXiv network error: {e}")
        except Exception as e:
            return self._error(ref, f"arXiv error: {e}")

    def _extract_match_with_authors(self, ref: Reference, entry: str) -> Tuple[MatchDetail, List[str]]:
        """Extract match details and author name list from arXiv XML entry."""
        # Title
        title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        matched_title = title_m.group(1).strip().replace("\n", " ") if title_m else ""
        title_sim = self.text_similarity(ref.title, matched_title)

        # Authors
        author_matches = re.findall(r"<name>(.*?)</name>", entry)
        matched_authors_str = ", ".join(author_matches[:3])
        if len(author_matches) > 3:
            matched_authors_str += " et al."
        author_sim = self.author_similarity(ref.authors, author_matches) if ref.authors else 1.0

        # Year
        pub_m = re.search(r"<published>(\d{4})", entry)
        matched_year = pub_m.group(1) if pub_m else ""
        year_match = ref.year == matched_year if ref.year and matched_year else True

        detail = MatchDetail(
            title_match=title_sim,
            author_match=author_sim,
            year_match=year_match,
            matched_title=matched_title,
            matched_authors=matched_authors_str,
            matched_year=matched_year,
        )

        return detail, author_matches

    def _extract_match(self, ref: Reference, entry: str) -> MatchDetail:
        """Extract match details from arXiv XML entry."""
        detail, _ = self._extract_match_with_authors(ref, entry)
        return detail

    def close(self) -> None:
        if self.client:
            self.client.close()
