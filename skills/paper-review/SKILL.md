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
| Model name | No | `gpt-5.2`, `anthropic/claude-opus-4-6`, `dashscope/qwen-vl-plus`. See **Model selection** below |
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
| `--base_url` | — | Custom API endpoint — must end at `/v1`, **not** `/v1/chat/completions` (litellm appends the path automatically) |
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
| `--vision` | **on** | Use vision mode (requires pypdfium2); enabled by default |
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

## Model selection

This pipeline uses [litellm](https://docs.litellm.ai/docs/providers) for model calls.
Provider prefixes are handled automatically by the pipeline — see the table below.

**IMPORTANT: The model MUST support multimodal (vision) input.** PDF review uses vision mode
(`--vision`) to render pages as images, which requires a vision-capable model. Text-only models
will fail or produce empty reviews.

The `--model` value uses a `provider/model-name` convention so the pipeline knows
which API endpoint to call.  The table below shows the exact string to pass:

| Provider | `--model` value | Env var | Notes |
|----------|----------------|---------|-------|
| OpenAI | `gpt-5.2`, `gpt-5-mini`, … | `OPENAI_API_KEY` | No prefix needed; `gpt-5.2` is the current flagship vision model; check [OpenAI models](https://platform.openai.com/docs/models) for the latest |
| Anthropic | `anthropic/claude-opus-4-6`, `anthropic/claude-sonnet-4-6`, … | `ANTHROPIC_API_KEY` | Use `anthropic/` prefix; `claude-opus-4-6` is the current flagship; check [Anthropic models](https://docs.anthropic.com/en/docs/about-claude/models) for the latest |
| DashScope (Qwen) | `dashscope/qwen-vl-plus`, `dashscope/qwen-vl-max`, … | `DASHSCOPE_API_KEY` | Use `dashscope/` prefix; the pipeline auto-routes to DashScope’s OpenAI-compatible endpoint |
| Custom endpoint | bare model name | `--api_key` + `--base_url` | Use the model name your endpoint expects; no prefix needed when `--base_url` is set |

> **Note on prefixes**: The `dashscope/` and `anthropic/` prefixes are interpreted by
> the pipeline itself — do **not** add them to the actual API key or base URL.
> For OpenAI models the bare model name (e.g. `gpt-5.2`) is sufficient.

**If the user does not specify a model**, choose one based on available API keys:
1. `DASHSCOPE_API_KEY` set → use `dashscope/qwen-vl-plus` (vision-capable)
2. `OPENAI_API_KEY` set → search web for the latest vision-capable OpenAI model and use it (currently `gpt-5.2`)
3. `ANTHROPIC_API_KEY` set → search web for the latest vision-capable Anthropic model and use it with `anthropic/` prefix (currently `anthropic/claude-opus-4-6`)

**Vision mode is enabled by default for PDF review.** Pages are rendered as images, which
preserves formatting, figures, and tables. To disable, pass `--no_vision` (not recommended).
The model **must** support multimodal (vision) input.

## Additional resources

- Full `PipelineConfig` options: [reference.md](reference.md)
- Discipline details and venues: [reference.md](reference.md#disciplines)

## Troubleshooting API errors

**CRITICAL: When the pipeline fails with an API error, you MUST diagnose and fix the root cause.
Do NOT fall back to reading the PDF as plain text yourself and calling the API manually —
this bypasses the entire review pipeline and produces incorrect, incomplete results.**

Diagnose by reading the full error message, then follow the checklist below:

### AuthenticationError / 401
- The API key is wrong or not set.
- Check the correct env var for the provider (see **Model selection** table).
- For DashScope: `echo $DASHSCOPE_API_KEY` — must be non-empty.
- Fix: export the correct key and re-run.

### NotFoundError / 404 — model not found
- The model name string is wrong.
- Search the web for the provider's current model list and use the exact API ID.
- Common mistakes: using a ChatGPT UI name instead of the API ID, outdated snapshot suffix.
- Fix: correct `--model` and re-run.

### BadRequestError / 400
- Often caused by `--base_url` ending with `/v1/chat/completions` instead of `/v1`.
  litellm appends the path automatically — strip everything after `/v1`.
- May also indicate the model does not support vision/image input.
  Use a vision-capable model (see **Model selection**) or omit `--vision`.
- Fix: correct `--base_url` or switch to a vision-capable model and re-run.

### Connection error / endpoint not reachable
- `--base_url` points to the wrong host or port.
- Test the endpoint first: `curl <base_url>/models -H "Authorization: Bearer <key>"`
- Fix: correct `--base_url` to the reachable endpoint and re-run.

### Timeout
- The model is taking too long (common for long PDFs with vision mode).
- Fix: increase `--timeout` (default 7500 s) or reduce `--vision_max_pages`.

### After fixing, always re-run the full pipeline command.
Never summarise or interpret the paper yourself as a substitute for a failed pipeline run.
