# -*- coding: utf-8 -*-
"""PubMed E-utilities verification for biomedical references."""

import re
from typing import List, Optional, Tuple

import httpx

from cookbooks.ref_hallucination_arena.schema import (
    MatchDetail,
    Reference,
    VerificationResult,
    VerificationStatus,
)
from cookbooks.ref_hallucination_arena.verifiers.base_verifier import BaseVerifier


class PubmedVerifier(BaseVerifier):
    """Verify references via PubMed E-utilities API.

    Uses esearch + esummary to find and match biomedical literature.
    Particularly effective for medical, biological, and clinical papers.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key
        headers = {"User-Agent": "RefArena/1.0"}
        self.client = httpx.Client(timeout=timeout, headers=headers)

    @property
    def source_name(self) -> str:
        return "pubmed"

    def verify(self, ref: Reference) -> VerificationResult:
        """Verify via PMID direct lookup or title search."""
        try:
            # 1. Direct PMID lookup
            if ref.pmid:
                return self._verify_by_pmid(ref, ref.pmid)

            # 2. Title search
            return self._verify_by_title(ref)
        except httpx.RequestError as e:
            return self._error(ref, f"PubMed network error: {e}")
        except Exception as e:
            return self._error(ref, f"PubMed error: {e}")

    def _verify_by_pmid(self, ref: Reference, pmid: str) -> VerificationResult:
        """Verify by direct PMID lookup with strict validation."""
        summaries = self._fetch_summaries([pmid])
        if not summaries:
            return self._not_found(ref, f"PMID {pmid} not found")

        info = summaries[0]
        detail, author_names = self._extract_match_with_authors(ref, info)

        # Strict validation: title + year + all provided authors
        if self.strict_match_with_author_list(ref, detail, author_names):
            score = self.compute_overall_score(detail)
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                confidence=score,
                message=self.format_match_message(detail, "PubMed PMID"),
                source=self.source_name,
                match_detail=detail,
                match_data=info,
            )

        score = self.compute_overall_score(detail)
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.NOT_FOUND,
            confidence=score,
            message=self.format_match_message(detail, "PubMed PMID mismatch"),
            match_detail=detail,
        )

    def _verify_by_title(self, ref: Reference) -> VerificationResult:
        """Search PubMed by title and verify with strict validation."""
        pmids = self._search_by_title(ref.title)
        if not pmids:
            return self._not_found(ref, "Not found in PubMed")

        summaries = self._fetch_summaries(pmids[:5])
        if not summaries:
            return self._not_found(ref, "PubMed summary fetch failed")

        best_verified: Optional[Tuple[dict, MatchDetail, float]] = None
        best_partial: Optional[Tuple[dict, MatchDetail, float]] = None

        for info in summaries:
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
                message=self.format_match_message(detail, "PubMed"),
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
                message=self.format_match_message(detail, "PubMed partial"),
                match_detail=detail,
            )

        return self._not_found(ref, "Not found in PubMed")

    def _search_by_title(self, title: str) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": f"{title}[Title]",
            "retmax": 5,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = self.client.get(self.ESEARCH_URL, params=params)
        if resp.status_code != 200:
            return []

        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_summaries(self, pmids: List[str]) -> List[dict]:
        """Fetch summary info for given PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = self.client.get(self.ESUMMARY_URL, params=params)
        if resp.status_code != 200:
            return []

        result = resp.json().get("result", {})
        summaries = []
        for pmid in pmids:
            if pmid in result:
                summaries.append(result[pmid])
        return summaries

    def _extract_match_with_authors(self, ref: Reference, info: dict) -> Tuple[MatchDetail, List[str]]:
        """Extract match details and author name list from PubMed summary."""
        # Title
        matched_title = info.get("title", "").rstrip(".")
        title_sim = self.text_similarity(ref.title, matched_title)

        # Authors
        authors_list = info.get("authors", [])
        author_names = [a.get("name", "") for a in authors_list if isinstance(a, dict)]
        matched_authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            matched_authors_str += " et al."
        author_sim = self.author_similarity(ref.authors, author_names) if ref.authors else 1.0

        # Year (from pubdate or epubdate)
        pub_date = info.get("pubdate", "") or info.get("epubdate", "")
        matched_year = ""
        year_match_re = re.search(r"(\d{4})", pub_date)
        if year_match_re:
            matched_year = year_match_re.group(1)
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
        """Extract match details from PubMed summary."""
        detail, _ = self._extract_match_with_authors(ref, info)
        return detail

    def close(self) -> None:
        if self.client:
            self.client.close()
