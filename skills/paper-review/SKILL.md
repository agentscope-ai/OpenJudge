---
name: paper-review
description: >
  Review academic papers for correctness, quality, and novelty using OpenJudge's
  multi-stage pipeline. Supports PDF files and LaTeX source packages (.tar.gz/.zip).
  Covers 10 disciplines: cs, medicine, physics, chemistry, biology, economics,
  psychology, environmental_science, mathematics, social_sciences.
  Use when the user asks to review, evaluate, critique, or assess a research paper,
  check references, or verify a BibTeX file.
---

# Paper Review Skill

Multi-stage academic paper review using the OpenJudge `PaperReviewPipeline`:

1. **Safety check** — jailbreak detection + format validation
2. **Correctness** — objective errors (math, logic, data inconsistencies)
3. **Review** — quality, novelty, significance (score 1–6)
4. **Criticality** — severity of correctness issues
5. **BibTeX verification** — cross-checks references against CrossRef/arXiv/DBLP

## Prerequisites

```bash
# Install OpenJudge
pip install py-openjudge

# Extra dependency for paper_review
pip install litellm
pip install pypdfium2  # only if using vision mode (use_vision_for_pdf=True)
```

## Gather from user before running

| Info | Required? | Notes |
|------|-----------|-------|
| Paper file path | Yes | PDF or .tar.gz/.zip TeX package |
| API key | Yes | Env var preferred: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. |
| Model name | No | Default: `gpt-4o`. Claude: `claude-opus-4-5`, Qwen: `qwen-plus` |
| Discipline | No | If not given, uses general CS/ML-oriented prompts |
| Venue | No | e.g. `"NeurIPS 2025"`, `"The Lancet"` |
| Instructions | No | Free-form reviewer guidance, e.g. `"Focus on experimental design"` |
| Language | No | `"en"` (default) or `"zh"` for Simplified Chinese output |
| BibTeX file | No | Required only for reference verification |
| CrossRef email | No | Improves API rate limits for BibTeX verification |

## Quick start

File type is auto-detected: `.pdf` → PDF review, `.tar.gz`/`.zip` → TeX review, `.bib` → BibTeX verification.

```bash
# Basic PDF review
python -m cookbooks.paper_review paper.pdf

# With discipline and venue
python -m cookbooks.paper_review paper.pdf \
  --discipline cs --venue "NeurIPS 2025"

# Chinese output
python -m cookbooks.paper_review paper.pdf --language zh

# Custom reviewer instructions
python -m cookbooks.paper_review paper.pdf \
  --instructions "Focus on experimental design and reproducibility"

# PDF + BibTeX verification
python -m cookbooks.paper_review paper.pdf \
  --bib references.bib --email your@email.com

# Vision mode (for models that prefer images over text extraction)
python -m cookbooks.paper_review paper.pdf \
  --vision --vision_max_pages 30 --format_vision_max_pages 10

# TeX source package
python -m cookbooks.paper_review paper_source.tar.gz \
  --discipline biology --email your@email.com

# TeX source package with Chinese output and custom instructions
python -m cookbooks.paper_review paper_source.tar.gz \
  --language zh --instructions "This is a short paper, be concise"

# Verify a standalone BibTeX file
python -m cookbooks.paper_review --bib_only references.bib --email your@email.com
```

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | — | Path to PDF, TeX package, or .bib file |
| `--bib_only` | — | Path to .bib file for standalone verification (no review) |
| `--model` | `gpt-4o` | Model name |
| `--api_key` | env var | API key |
| `--base_url` | — | Custom API endpoint |
| `--discipline` | — | Academic discipline |
| `--venue` | — | Target conference/journal |
| `--instructions` | — | Free-form reviewer guidance |
| `--language` | `en` | Output language: `en` or `zh` |
| `--bib` | — | Path to .bib file (for PDF review + reference verification) |
| `--email` | — | CrossRef mailto for BibTeX check |
| `--paper_name` | filename stem | Paper title in report |
| `--output` | auto | Output .md report path |
| `--no_safety` | off | Skip safety checks |
| `--no_correctness` | off | Skip correctness check |
| `--no_criticality` | off | Skip criticality verification |
| `--no_bib` | off | Skip BibTeX verification |
| `--vision` | off | Use vision mode (requires pypdfium2) |
| `--vision_max_pages` | `30` | Max pages in vision mode (0 = all) |
| `--format_vision_max_pages` | `10` | Max pages for format check (0 = use `--vision_max_pages`) |
| `--timeout` | `7500` | API timeout in seconds |

## Interpreting results

**Review score (1–6):**
- 1–2: Reject (major flaws or well-known results)
- 3: Borderline reject
- 4: Borderline accept
- 5–6: Accept / Strong accept

**Correctness score (1–3):**
- 1: No objective errors
- 2: Minor errors (notation, arithmetic in non-critical parts)
- 3: Major errors (wrong proofs, core algorithm flaws)

**BibTeX verification:**
- `verified`: found in CrossRef/arXiv/DBLP
- `suspect`: title/author mismatch or not found — manual check recommended

## API key by model

| Model prefix | Environment variable |
|-------------|---------------------|
| `gpt-*`, `o1-*`, `o3-*` | `OPENAI_API_KEY` |
| `claude-*` | `ANTHROPIC_API_KEY` |
| `qwen-*`, `dashscope/*` | `DASHSCOPE_API_KEY` |
| Custom endpoint | `--api-key` + `--base-url` |

## Additional resources

- Full `PipelineConfig` options: [reference.md](reference.md)
- Discipline details and venues: [reference.md](reference.md#disciplines)
