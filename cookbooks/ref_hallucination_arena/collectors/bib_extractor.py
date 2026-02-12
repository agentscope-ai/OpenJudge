# -*- coding: utf-8 -*-
"""Extract BibTeX references from free-text model responses."""

import re
from typing import List, Optional

from cookbooks.ref_hallucination_arena.schema import Reference


class BibExtractor:
    """Extract BibTeX entries from model responses.

    Strategies (tried in order):
      1. Extract content inside ```bib / ```bibtex code fences.
      2. Extract standalone @type{...} entries scattered in the text.
      3. Fallback: try to parse structured plain-text references.
    """

    # Matches ```bib or ```bibtex fenced code blocks
    _FENCE_PATTERN = re.compile(
        r"```(?:bib(?:tex)?)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    # Matches a full BibTeX entry: @type{key, ... }
    # Uses brace-counting to handle nested braces correctly
    _ENTRY_START_PATTERN = re.compile(
        r"@(\w+)\s*\{\s*([^,\s]*)\s*,",
        re.IGNORECASE,
    )

    def extract(self, response_text: str) -> List[Reference]:
        """Extract references from a model response.

        Args:
            response_text: Raw text response from the model.

        Returns:
            List of extracted Reference objects.
        """
        if not response_text:
            return []

        # Strategy 1: fenced code blocks
        fenced_content = self._extract_fenced(response_text)
        if fenced_content:
            refs = self._parse_bibtex(fenced_content)
            if refs:
                return refs

        # Strategy 2: standalone entries in text
        refs = self._parse_bibtex(response_text)
        if refs:
            return refs

        # Strategy 3: plain-text fallback (numbered references)
        return self._parse_plain_text(response_text)

    def _extract_fenced(self, text: str) -> str:
        """Extract content from ```bib/bibtex fenced blocks."""
        blocks = self._FENCE_PATTERN.findall(text)
        if blocks:
            return "\n\n".join(blocks)
        return ""

    def _parse_bibtex(self, text: str) -> List[Reference]:
        """Parse BibTeX entries using brace-counting for robustness."""
        refs = []

        for match in self._ENTRY_START_PATTERN.finditer(text):
            entry_type = match.group(1).lower()
            key = match.group(2).strip()

            # Find the matching closing brace via counting
            start = match.start()
            brace_start = text.index("{", start)
            fields_str = self._extract_braced_content(text, brace_start)
            if fields_str is None:
                continue

            ref = self._parse_fields(key, entry_type, fields_str)
            if ref:
                refs.append(ref)

        return refs

    def _extract_braced_content(self, text: str, open_pos: int) -> Optional[str]:
        """Extract content between matched braces starting at open_pos."""
        depth = 0
        for i in range(open_pos, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[open_pos + 1 : i]
        return None  # unmatched

    def _parse_fields(self, key: str, entry_type: str, fields_str: str) -> Optional[Reference]:
        """Parse individual fields from BibTeX entry body."""

        def extract_field(name: str) -> Optional[str]:
            # Match field = {value}, field = "value", or field = number
            # Try brace-delimited value first (handles nested braces)
            brace_pattern = rf"{name}\s*=\s*\{{(.*?)\}}"
            m = re.search(brace_pattern, fields_str, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
            # Try quote-delimited value
            quote_pattern = rf'{name}\s*=\s*"(.*?)"'
            m = re.search(quote_pattern, fields_str, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
            # Try unquoted numeric value (e.g., year = 2023)
            num_pattern = rf"{name}\s*=\s*(\d+)"
            m = re.search(num_pattern, fields_str, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            return None

        title = extract_field("title")
        if not title:
            return None

        # Extract arXiv ID
        arxiv_id = None
        journal = extract_field("journal") or extract_field("booktitle") or ""
        eprint = extract_field("eprint")
        if eprint:
            arxiv_id = eprint
        elif "arxiv" in journal.lower():
            arxiv_match = re.search(r"(\d{4}\.\d{4,5})", journal)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)

        # Extract PMID from note or url
        pmid = None
        note = extract_field("note") or ""
        url = extract_field("url") or ""
        pmid_match = re.search(r"(?:PMID|pmid)[:\s]*(\d+)", note + " " + url)
        if pmid_match:
            pmid = pmid_match.group(1)

        return Reference(
            key=key,
            title=title,
            authors=extract_field("author"),
            year=extract_field("year"),
            journal=journal,
            doi=extract_field("doi"),
            arxiv_id=arxiv_id,
            pmid=pmid,
            entry_type=entry_type,
        )

    def _parse_plain_text(self, text: str) -> List[Reference]:
        """Fallback: parse numbered plain-text references.

        Handles patterns like:
          1. Author et al. (2023). "Title". Journal.
          [1] Author et al., "Title", Journal, 2023.
        """
        refs = []

        # Pattern: numbered reference with quoted title
        patterns = [
            # "1. Authors (Year). Title. Journal."
            re.compile(
                r"(?:^|\n)\s*(?:\d+[\.\)]\s*|[\[\(]\d+[\]\)]\s*)"
                r"(.+?)\s*[\(\[]?(\d{4})[\)\]]?\s*[\.\,]\s*"
                r'["\u201c](.+?)["\u201d]',
                re.MULTILINE,
            ),
            # Simpler: "Title" (Year)
            re.compile(
                r'["\u201c](.+?)["\u201d]\s*[\(\[]?(\d{4})[\)\]]?',
            ),
        ]

        seen_titles = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                groups = m.groups()
                if len(groups) >= 3:
                    authors, year, title = groups[0], groups[1], groups[2]
                elif len(groups) >= 2:
                    title, year = groups[0], groups[1]
                    authors = None
                else:
                    continue

                title_lower = title.strip().lower()
                if title_lower in seen_titles or len(title_lower) < 10:
                    continue
                seen_titles.add(title_lower)

                refs.append(
                    Reference(
                        key=f"ref_{len(refs)+1}",
                        title=title.strip(),
                        authors=authors.strip() if authors else None,
                        year=year.strip(),
                    )
                )

        return refs
