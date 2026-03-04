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

    # Patterns for stripping thinking / reasoning blocks that some models
    # embed directly in the content field (e.g. DeepSeek, QwQ, Kimi, etc.)
    _THINKING_PATTERNS = [
        # <think>...</think>  (most common)
        re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
        # <thinking>...</thinking>
        re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
        # <reasoning>...</reasoning>
        re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
        # <reflection>...</reflection>
        re.compile(r"<reflection>.*?</reflection>", re.DOTALL | re.IGNORECASE),
    ]

    @classmethod
    def _strip_thinking(cls, text: str) -> str:
        """Remove thinking / reasoning blocks from model output.

        Some models (DeepSeek, QwQ, Kimi, Grok, etc.) may embed their
        chain-of-thought reasoning inside ``<think>…</think>`` or similar
        tags directly in the ``content`` field.  These blocks can contain
        partial BibTeX-like text that would confuse the extractor.
        """
        for pattern in cls._THINKING_PATTERNS:
            text = pattern.sub("", text)
        return text.strip()

    def extract(self, response_text: str) -> List[Reference]:
        """Extract references from a model response.

        Args:
            response_text: Raw text response from the model.

        Returns:
            List of extracted Reference objects.
        """
        if not response_text:
            return []

        # Strip any thinking / reasoning blocks before extraction
        response_text = self._strip_thinking(response_text)
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

    @staticmethod
    def _strip_latex(text: str) -> str:
        r"""Strip LaTeX markup from a BibTeX field value.

        Handles (in order):
          - Accent commands with braces: {\"{o}} → o, {\v{Z}} → Z
          - Accent shorthand with braces: \'{e} → e, \"{o} → o
          - Accent shorthand without braces: \'e → e, \"o → o
          - Other LaTeX commands: \textbf{X} → X, \emph{Y} → Y
          - Math mode: {$L$} → L, $T_c$ → Tc
          - Remaining braces: {CAR-T} → CAR-T
          - Tildes used as non-breaking spaces: ~ → space
        """
        # Pass 1: {\cmd{X}} → X  (e.g. {\"o} → o, {\v{Z}} → Z)
        text = re.sub(r"\{\\[^a-zA-Z]?\{([^}]*)\}\}", r"\1", text)
        # Pass 2: \cmd{X} → X  (e.g. \textbf{word}, \v{Z}, \"{o})
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
        # Pass 3: \'X or \"X  (accent shorthand with single char, no braces)
        text = re.sub(r"\\['\"`^~=.uvHtcdbkr]\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\['\"`^~=.uvHtcdbkr]([a-zA-Z])", r"\1", text)
        # Pass 4: remaining backslash commands like \& → &
        text = re.sub(r"\\([&%#])", r"\1", text)
        # Pass 5: any leftover \command → remove
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        # Strip $ (math mode delimiters) and _ ^ (sub/superscripts)
        text = re.sub(r"[$_^]", "", text)
        # Remove remaining braces
        text = re.sub(r"[{}]", "", text)
        # Replace ~ (non-breaking space) with regular space
        text = text.replace("~", " ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_fields(self, key: str, entry_type: str, fields_str: str) -> Optional[Reference]:
        """Parse individual fields from BibTeX entry body."""

        def extract_field(name: str) -> Optional[str]:
            """Extract a BibTeX field value, correctly handling nested braces.

            Strategy:
              1. Find ``name = {`` and use brace-counting to extract the full
                 value, including any nested ``{...}`` groups.
              2. Fall back to quote-delimited: ``name = "..."``
              3. Fall back to bare numeric: ``name = 2023``
            """
            # --- Strategy 1: brace-delimited with depth counting ---
            pattern = re.compile(rf"{name}\s*=\s*\{{", re.IGNORECASE)
            m = pattern.search(fields_str)
            if m:
                # Position of the opening brace
                open_pos = m.end() - 1  # points at '{'
                value = _extract_braced_value(fields_str, open_pos)
                if value is not None:
                    return value.strip()

            # --- Strategy 2: quote-delimited ---
            quote_pattern = rf'{name}\s*=\s*"(.*?)"'
            m = re.search(quote_pattern, fields_str, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()

            # --- Strategy 3: bare numeric (e.g. year = 2023) ---
            num_pattern = rf"{name}\s*=\s*(\d+)"
            m = re.search(num_pattern, fields_str, re.IGNORECASE)
            if m:
                return m.group(1).strip()

            return None

        def _extract_braced_value(text: str, open_pos: int) -> Optional[str]:
            """Extract content between matched braces using depth counting."""
            depth = 0
            for i in range(open_pos, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[open_pos + 1 : i]
            return None

        title_raw = extract_field("title")
        if not title_raw:
            return None

        # Clean LaTeX markup from title and authors so that downstream
        # verification can match against plain-text database records.
        title = self._strip_latex(title_raw)
        authors_raw = extract_field("author")
        authors = self._strip_latex(authors_raw) if authors_raw else None

        # Extract arXiv ID
        arxiv_id = None
        journal = extract_field("journal") or extract_field("booktitle") or ""
        eprint = extract_field("eprint")
        if eprint:
            arxiv_id = self._strip_latex(eprint)
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
            authors=authors,
            year=extract_field("year"),
            journal=self._strip_latex(journal) if journal else "",
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
