---
name: bib-verify
description: >
  Verify a BibTeX file for hallucinated or fabricated references by cross-checking
  every entry against CrossRef, arXiv, and DBLP. Reports each reference as
  verified, suspect, or not found, with field-level mismatch details (title,
  authors, year, DOI). Use when the user wants to check a .bib file for fake
  citations, validate references in a paper, or audit bibliography entries for
  accuracy.
---

# BibTeX Verification Skill

Check every entry in a `.bib` file against real academic databases using the
OpenJudge `PaperReviewPipeline` in BibTeX-only mode:

1. **Parse** — extract all entries from the `.bib` file
2. **Lookup** — query CrossRef, arXiv, and DBLP for each reference
3. **Match** — compare title, authors, year, and DOI
4. **Report** — flag each entry as `verified`, `suspect`, or `not_found`

## Prerequisites

```bash
pip install py-openjudge litellm
```

## Gather from user before running

| Info | Required? | Notes |
|------|-----------|-------|
| BibTeX file path | Yes | `.bib` file to verify |
| CrossRef email | No | Improves CrossRef API rate limits |

## Quick start

```bash
# Verify a standalone .bib file
python -m cookbooks.paper_review --bib_only references.bib

# With CrossRef email for better rate limits
python -m cookbooks.paper_review --bib_only references.bib --email your@email.com

# Save report to a custom path
python -m cookbooks.paper_review --bib_only references.bib \
  --email your@email.com --output bib_report.md
```

## Relevant options

| Flag | Default | Description |
|------|---------|-------------|
| `--bib_only` | — | Path to `.bib` file (required for standalone verification) |
| `--email` | — | CrossRef mailto — improves rate limits, recommended |
| `--output` | auto | Output `.md` report path |
| `--language` | `en` | Report language: `en` or `zh` |

## Interpreting results

Each reference entry is assigned one of three statuses:

| Status | Meaning |
|--------|---------|
| `verified` | Found in CrossRef / arXiv / DBLP with matching fields |
| `suspect` | Title or authors do not match any real paper — likely hallucinated or mis-cited |
| `not_found` | No match in any database — treat as fabricated |

**Field-level details** are shown for `suspect` entries:
- `title_match` — whether the title matches a real paper
- `author_match` — whether the author list matches
- `year_match` — whether the publication year is correct
- `doi_match` — whether the DOI resolves to the right paper

## Additional resources

- Full pipeline options: [../paper-review/reference.md](../paper-review/reference.md)
- Combined PDF review + BibTeX verification: [../paper-review/SKILL.md](../paper-review/SKILL.md)
