# -*- coding: utf-8 -*-
"""BibTeX reference verification using Crossref, arXiv, and DBLP APIs."""

import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from cookbooks.paper_review.schema import VerificationStatus


@dataclass
class Reference:
    """A single reference entry."""

    key: str
    title: str
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


@dataclass
class MatchDetail:
    """Detailed matching information."""

    title_match: float = 0.0
    author_match: float = 0.0
    year_match: bool = False
    matched_title: str = ""
    matched_authors: str = ""
    matched_year: str = ""


@dataclass
class VerificationResult:
    """Result of verifying a single reference."""

    reference: Reference
    status: VerificationStatus
    confidence: float = 0.0
    message: str = ""
    source: str = ""
    match_detail: Optional[MatchDetail] = None
    match_data: Optional[Dict[str, Any]] = None


class BibChecker:
    """Verify references using Crossref, arXiv, and DBLP APIs.

    Checks title, authors, and year for comprehensive verification.
    """

    CROSSREF_API = "https://api.crossref.org/works"
    ARXIV_API = "https://export.arxiv.org/api/query"
    DBLP_API = "https://dblp.org/search/publ/api"

    def __init__(self, mailto: Optional[str] = None, timeout: float = 30.0):
        self.mailto = mailto
        headers = {"User-Agent": f"BibChecker/1.0 (mailto:{mailto})" if mailto else "BibChecker/1.0"}
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def parse_bib_file(self, bib_content: str) -> List[Reference]:
        """Parse BibTeX content and extract references."""
        references = []
        entry_pattern = r"@(\w+)\s*\{\s*([^,]+)\s*,([^@]*)}"
        entries = re.findall(entry_pattern, bib_content, re.DOTALL)

        for _, key, fields in entries:
            ref = self._parse_entry(key.strip(), fields)
            if ref:
                references.append(ref)
        return references

    def _parse_entry(self, key: str, fields: str) -> Optional[Reference]:
        """Parse a single BibTeX entry."""

        def extract_field(name: str) -> Optional[str]:
            pattern = rf'{name}\s*=\s*[{{"](.*?)[}}"]'
            match = re.search(pattern, fields, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else None

        title = extract_field("title")
        if not title:
            return None

        arxiv_id = None
        journal = extract_field("journal") or extract_field("booktitle") or ""
        eprint = extract_field("eprint")

        if eprint:
            arxiv_id = eprint
        elif "arxiv" in journal.lower():
            arxiv_match = re.search(r"(\d{4}\.\d{4,5})", journal)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)

        return Reference(
            key=key,
            title=title,
            authors=extract_field("author"),
            year=extract_field("year"),
            journal=journal,
            doi=extract_field("doi"),
            arxiv_id=arxiv_id,
        )

    def verify_reference(self, ref: Reference) -> VerificationResult:
        """Verify a single reference using multiple sources."""
        try:
            # 1. DOI (most reliable)
            if ref.doi:
                result = self._verify_crossref_doi(ref)
                if result.status == VerificationStatus.VERIFIED:
                    return result

            # 2. arXiv
            if ref.arxiv_id or (ref.journal and "arxiv" in ref.journal.lower()):
                result = self._verify_arxiv(ref)
                if result.status == VerificationStatus.VERIFIED:
                    return result

            # 3. DBLP
            result = self._verify_dblp(ref)
            if result.status == VerificationStatus.VERIFIED:
                return result

            # 4. Crossref title search
            return self._verify_crossref_title(ref)

        except httpx.RequestError as e:
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.ERROR,
                message=f"Network error: {str(e)}",
            )
        except (KeyError, ValueError, TypeError) as e:
            return VerificationResult(
                reference=ref,
                status=VerificationStatus.ERROR,
                message=f"Data parsing error: {str(e)}",
            )

    def _verify_crossref_doi(self, ref: Reference) -> VerificationResult:
        """Verify by DOI via Crossref API."""
        try:
            url = f"{self.CROSSREF_API}/{urllib.parse.quote(ref.doi, safe='')}"
            resp = self.client.get(url)
            if resp.status_code == 200:
                data = resp.json().get("message", {})
                detail = self._extract_crossref_match(ref, data)
                return VerificationResult(
                    reference=ref,
                    status=VerificationStatus.VERIFIED,
                    confidence=1.0,
                    message=self._format_match_message(detail, "DOI verified"),
                    source="crossref",
                    match_detail=detail,
                    match_data=data,
                )
        except Exception:
            pass
        return VerificationResult(
            reference=ref,
            status=VerificationStatus.SUSPECT,
            message="DOI not found",
        )

    def _verify_crossref_title(self, ref: Reference) -> VerificationResult:
        """Verify by title search via Crossref API."""
        try:
            params = {"query.title": ref.title, "rows": 5}
            if self.mailto:
                params["mailto"] = self.mailto
            resp = self.client.get(self.CROSSREF_API, params=params)

            if resp.status_code != 200:
                return VerificationResult(
                    reference=ref, status=VerificationStatus.SUSPECT, message="Crossref API error"
                )

            items = resp.json().get("message", {}).get("items", [])
            if not items:
                return VerificationResult(
                    reference=ref, status=VerificationStatus.SUSPECT, message="Not found in Crossref"
                )

            # Find best match considering title + author + year
            best_result = None
            best_score = 0

            for item in items:
                detail = self._extract_crossref_match(ref, item)
                score = self._compute_overall_score(detail)
                if score > best_score:
                    best_score = score
                    best_result = (item, detail)

            if best_result:
                item, detail = best_result
                if best_score >= 0.7:
                    return VerificationResult(
                        reference=ref,
                        status=VerificationStatus.VERIFIED,
                        confidence=best_score,
                        message=self._format_match_message(detail, "Crossref"),
                        source="crossref",
                        match_detail=detail,
                        match_data=item,
                    )
                elif best_score >= 0.4:
                    return VerificationResult(
                        reference=ref,
                        status=VerificationStatus.SUSPECT,
                        confidence=best_score,
                        message=self._format_match_message(detail, "Partial"),
                        source="crossref",
                        match_detail=detail,
                    )

            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="Low similarity")
        except Exception as e:
            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message=f"Crossref error: {e}")

    def _extract_crossref_match(self, ref: Reference, data: dict) -> MatchDetail:
        """Extract match details from Crossref response."""
        # Title
        titles = data.get("title", [])
        matched_title = titles[0] if titles else ""
        title_sim = self._text_similarity(ref.title, matched_title)

        # Authors
        authors = data.get("author", [])
        author_names = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors]
        matched_authors = ", ".join(author_names[:3])
        if len(author_names) > 3:
            matched_authors += " et al."
        author_sim = self._author_similarity(ref.authors, author_names) if ref.authors else 1.0

        # Year
        pub_date = data.get("published-print") or data.get("published-online") or data.get("created")
        matched_year = ""
        if pub_date and "date-parts" in pub_date:
            parts = pub_date["date-parts"][0]
            matched_year = str(parts[0]) if parts else ""
        year_match = ref.year == matched_year if ref.year and matched_year else True

        return MatchDetail(
            title_match=title_sim,
            author_match=author_sim,
            year_match=year_match,
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_year=matched_year,
        )

    def _verify_arxiv(self, ref: Reference) -> VerificationResult:
        """Verify via arXiv API."""
        try:
            if ref.arxiv_id:
                query = f"id:{ref.arxiv_id}"
            else:
                query = f"ti:{urllib.parse.quote(ref.title)}"

            url = f"{self.ARXIV_API}?search_query={query}&max_results=5"
            resp = self.client.get(url)

            if resp.status_code != 200:
                return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="arXiv API error")

            # Parse XML
            entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
            if not entries:
                return VerificationResult(
                    reference=ref, status=VerificationStatus.SUSPECT, message="Not found on arXiv"
                )

            for entry in entries:
                detail = self._extract_arxiv_match(ref, entry)
                score = self._compute_overall_score(detail)
                if score >= 0.7:
                    return VerificationResult(
                        reference=ref,
                        status=VerificationStatus.VERIFIED,
                        confidence=score,
                        message=self._format_match_message(detail, "arXiv"),
                        source="arxiv",
                        match_detail=detail,
                    )

            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="Mismatch on arXiv")
        except Exception as e:
            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message=f"arXiv error: {e}")

    def _extract_arxiv_match(self, ref: Reference, entry: str) -> MatchDetail:
        """Extract match details from arXiv entry."""
        # Title
        title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        matched_title = title_match.group(1).strip().replace("\n", " ") if title_match else ""
        title_sim = self._text_similarity(ref.title, matched_title)

        # Authors
        author_matches = re.findall(r"<name>(.*?)</name>", entry)
        matched_authors = ", ".join(author_matches[:3])
        if len(author_matches) > 3:
            matched_authors += " et al."
        author_sim = self._author_similarity(ref.authors, author_matches) if ref.authors else 1.0

        # Year (from published date)
        pub_match = re.search(r"<published>(\d{4})", entry)
        matched_year = pub_match.group(1) if pub_match else ""
        year_match = ref.year == matched_year if ref.year and matched_year else True

        return MatchDetail(
            title_match=title_sim,
            author_match=author_sim,
            year_match=year_match,
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_year=matched_year,
        )

    def _verify_dblp(self, ref: Reference) -> VerificationResult:
        """Verify via DBLP API."""
        try:
            query = urllib.parse.quote(ref.title)
            url = f"{self.DBLP_API}?q={query}&format=json&h=5"
            resp = self.client.get(url)

            if resp.status_code != 200:
                return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="DBLP API error")

            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="Not found on DBLP")

            for hit in hits:
                info = hit.get("info", {})
                detail = self._extract_dblp_match(ref, info)
                score = self._compute_overall_score(detail)
                if score >= 0.7:
                    return VerificationResult(
                        reference=ref,
                        status=VerificationStatus.VERIFIED,
                        confidence=score,
                        message=self._format_match_message(detail, "DBLP"),
                        source="dblp",
                        match_detail=detail,
                        match_data=info,
                    )

            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message="Mismatch on DBLP")
        except Exception as e:
            return VerificationResult(reference=ref, status=VerificationStatus.SUSPECT, message=f"DBLP error: {e}")

    def _extract_dblp_match(self, ref: Reference, info: dict) -> MatchDetail:
        """Extract match details from DBLP response."""
        # Title
        matched_title = info.get("title", "").rstrip(".")
        title_sim = self._text_similarity(ref.title, matched_title)

        # Authors
        authors = info.get("authors", {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        author_names = [a.get("text", a) if isinstance(a, dict) else a for a in authors]
        matched_authors = ", ".join(author_names[:3])
        if len(author_names) > 3:
            matched_authors += " et al."
        author_sim = self._author_similarity(ref.authors, author_names) if ref.authors else 1.0

        # Year
        matched_year = info.get("year", "")
        year_match = ref.year == matched_year if ref.year and matched_year else True

        return MatchDetail(
            title_match=title_sim,
            author_match=author_sim,
            year_match=year_match,
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_year=matched_year,
        )

    def _text_similarity(self, t1: str, t2: str) -> float:
        """Calculate text similarity (Jaccard on words)."""
        s1 = set(re.sub(r"[^\w\s]", "", t1.lower()).split())
        s2 = set(re.sub(r"[^\w\s]", "", t2.lower()).split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _author_similarity(self, bib_authors: str, matched_authors: List[str]) -> float:
        """Calculate author similarity."""
        if not bib_authors or not matched_authors:
            return 0.0

        # Extract last names from bib authors (handles "LastName, FirstName and ..." format)
        bib_names = set()
        for part in re.split(r"\s+and\s+", bib_authors, flags=re.IGNORECASE):
            # Try "LastName, FirstName" format
            if "," in part:
                last = part.split(",")[0].strip().lower()
            else:
                # "FirstName LastName" format - take last word
                words = part.strip().split()
                last = words[-1].lower() if words else ""
            if last and last != "others":
                bib_names.add(last)

        # Extract last names from matched authors
        matched_names = set()
        for name in matched_authors:
            words = name.strip().split()
            if words:
                matched_names.add(words[-1].lower())

        if not bib_names or not matched_names:
            return 0.0

        # Jaccard similarity on last names
        intersection = len(bib_names & matched_names)
        union = len(bib_names | matched_names)
        return intersection / union if union > 0 else 0.0

    def _compute_overall_score(self, detail: MatchDetail) -> float:
        """Compute overall match score from detail."""
        # Weights: title 50%, author 30%, year 20%
        score = detail.title_match * 0.5 + detail.author_match * 0.3
        if detail.year_match:
            score += 0.2
        return score

    def _format_match_message(self, detail: MatchDetail, source: str) -> str:
        """Format match message with details."""
        parts = [f"T:{detail.title_match:.0%}", f"A:{detail.author_match:.0%}"]
        if detail.year_match:
            parts.append("Y:✓")
        else:
            parts.append("Y:✗")
        return f"{source} ({', '.join(parts)})"

    def verify_all(self, references: List[Reference], max_workers: int = 5) -> List[VerificationResult]:
        """Verify all references concurrently."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.verify_reference, ref): ref for ref in references}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def check_bib_file(self, bib_path: str) -> Dict[str, Any]:
        """Check all references in a .bib file."""
        with open(bib_path, "r", encoding="utf-8") as f:
            return self.check_bib_content(f.read())

    def check_bib_content(self, bib_content: str) -> Dict[str, Any]:
        """Check all references in BibTeX content string."""
        references = self.parse_bib_file(bib_content)
        results = self.verify_all(references)

        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        suspect = sum(1 for r in results if r.status == VerificationStatus.SUSPECT)
        errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)

        sources = {}
        for r in results:
            if r.status == VerificationStatus.VERIFIED and r.source:
                sources[r.source] = sources.get(r.source, 0) + 1

        return {
            "total_references": len(references),
            "verified": verified,
            "suspect": suspect,
            "errors": errors,
            "verification_rate": verified / len(references) if references else 0,
            "verified_by_source": sources,
            "results": results,
        }

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False

    def close(self):
        """Explicitly close the HTTP client."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            self.client = None
