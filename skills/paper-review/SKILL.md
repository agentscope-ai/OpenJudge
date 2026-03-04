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

# PDF + BibTeX verification
python skills/paper-review/scripts/review_paper.py paper.pdf \
  --bib references.bib --email your@email.com

# TeX source package
python skills/paper-review/scripts/review_tex.py paper_source.tar.gz \
  --discipline biology --email your@email.com
```

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
