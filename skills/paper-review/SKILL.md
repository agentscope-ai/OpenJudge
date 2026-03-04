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
# Install the project
git clone https://github.com/agentscope-ai/OpenJudge.git
cd OpenJudge
pip install -e .

# Extra dependencies for paper_review
pip install litellm httpx
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

## Choose the right script

| User has | Run |
|----------|-----|
| PDF only | `scripts/review_paper.py` |
| PDF + .bib | `scripts/review_paper.py` with `--bib` |
| TeX source package (.tar.gz / .zip) | `scripts/review_tex.py` |
| BibTeX file only | `scripts/review_tex.py --bib-only` |

## Quick start

```bash
# Basic PDF review
python skills/paper-review/scripts/review_paper.py paper.pdf

# With discipline and venue
python skills/paper-review/scripts/review_paper.py paper.pdf \
  --discipline cs --venue "NeurIPS 2025"

# Chinese output
python skills/paper-review/scripts/review_paper.py paper.pdf --language zh

# Custom reviewer instructions
python skills/paper-review/scripts/review_paper.py paper.pdf \
  --instructions "Focus on experimental design and reproducibility"

# PDF + BibTeX verification
python skills/paper-review/scripts/review_paper.py paper.pdf \
  --bib references.bib --email your@email.com

# Vision mode (for models that prefer images over text extraction)
python skills/paper-review/scripts/review_paper.py paper.pdf \
  --vision --vision-max-pages 30 --format-vision-max-pages 10

# TeX source package
python skills/paper-review/scripts/review_tex.py paper_source.tar.gz \
  --discipline biology --email your@email.com

# TeX source package with Chinese output and custom instructions
python skills/paper-review/scripts/review_tex.py paper_source.tar.gz \
  --language zh --instructions "This is a short paper, be concise"
```

## All options — review_paper.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o` | Model name |
| `--api-key` | env var | API key |
| `--base-url` | — | Custom API endpoint |
| `--discipline` | — | Academic discipline |
| `--venue` | — | Target conference/journal |
| `--instructions` | — | Free-form reviewer guidance |
| `--language` | `en` | Output language: `en` or `zh` |
| `--bib` | — | Path to .bib file |
| `--email` | — | CrossRef mailto for BibTeX check |
| `--paper-name` | PDF stem | Paper title in report |
| `--output` | auto | Output .md report path |
| `--no-safety` | off | Skip safety checks |
| `--no-correctness` | off | Skip correctness check |
| `--no-criticality` | off | Skip criticality verification |
| `--no-bib` | off | Skip BibTeX verification |
| `--vision` | off | Use vision mode (requires pypdfium2) |
| `--vision-max-pages` | `30` | Max pages in vision mode (0 = all) |
| `--format-vision-max-pages` | `10` | Max pages for format check (0 = use `--vision-max-pages`) |
| `--timeout` | `7500` | API timeout in seconds |

## All options — review_tex.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o` | Model name |
| `--api-key` | env var | API key |
| `--base-url` | — | Custom API endpoint |
| `--discipline` | — | Academic discipline |
| `--venue` | — | Target conference/journal |
| `--instructions` | — | Free-form reviewer guidance |
| `--language` | `en` | Output language: `en` or `zh` |
| `--email` | — | CrossRef mailto for BibTeX check |
| `--paper-name` | package stem | Paper title in report |
| `--output` | auto | Output .md report path |
| `--no-safety` | off | Skip safety checks |
| `--no-correctness` | off | Skip correctness check |
| `--no-criticality` | off | Skip criticality verification |
| `--no-bib` | off | Skip BibTeX verification |
| `--bib-only` | — | Verify a standalone .bib file only |
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
