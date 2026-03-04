---
name: ref-hallucination-arena
description: >
  Benchmark LLM reference recommendation capabilities by verifying every cited
  paper against Crossref, PubMed, arXiv, and DBLP. Measures hallucination rate,
  per-field accuracy (title/author/year/DOI), discipline breakdown, and year
  constraint compliance. Supports tool-augmented (ReAct + web search) mode.
  Use when the user asks to evaluate, benchmark, or compare models on academic
  reference hallucination, literature recommendation quality, or citation accuracy.
---

# Reference Hallucination Arena Skill

Evaluate how accurately LLMs recommend real academic references using the
OpenJudge `RefArenaPipeline`:

1. **Load queries** — from JSON/JSONL dataset
2. **Collect responses** — BibTeX-formatted references from target models
3. **Extract references** — parse BibTeX entries from model output
4. **Verify references** — cross-check against Crossref / PubMed / arXiv / DBLP
5. **Score & rank** — compute verification rate, per-field accuracy, discipline breakdown
6. **Generate report** — Markdown report + visualization charts

## Prerequisites

```bash
# Install OpenJudge
pip install py-openjudge

# Extra dependency for ref_hallucination_arena (chart generation)
pip install matplotlib
```

## Gather from user before running

| Info | Required? | Notes |
|------|-----------|-------|
| Config YAML path | Yes | Defines endpoints, dataset, verification settings |
| Dataset path | Yes | JSON/JSONL file with queries (can be set in config) |
| API keys | Yes | Env vars: `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, etc. |
| CrossRef email | No | Improves API rate limits for verification |
| PubMed API key | No | Improves PubMed rate limits |
| Output directory | No | Default: `./evaluation_results/ref_hallucination_arena` |
| Report language | No | `"en"` (default) or `"zh"` |
| Tavily API key | No | Required only if using tool-augmented mode |

## Quick start

### CLI

```bash
# Run evaluation with config file
python -m cookbooks.ref_hallucination_arena --config config.yaml --save

# Resume from checkpoint (default behavior)
python -m cookbooks.ref_hallucination_arena --config config.yaml --save

# Start fresh, ignore checkpoint
python -m cookbooks.ref_hallucination_arena --config config.yaml --fresh --save

# Override output directory
python -m cookbooks.ref_hallucination_arena --config config.yaml \
  --output_dir ./my_results --save
```

### Python API

```python
import asyncio
from cookbooks.ref_hallucination_arena.pipeline import RefArenaPipeline

async def main():
    pipeline = RefArenaPipeline.from_config("config.yaml")
    result = await pipeline.evaluate()

    for rank, (model, score) in enumerate(result.rankings, 1):
        print(f"{rank}. {model}: {score:.1%}")

asyncio.run(main())
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML configuration file (required) |
| `--output_dir` | config value | Override output directory |
| `--save` | `False` | Save results to file |
| `--fresh` | `False` | Start fresh, ignore checkpoint |

## Minimal config file

```yaml
task:
  description: "Evaluate LLM reference recommendation capabilities"

dataset:
  path: "./data/queries.json"

target_endpoints:
  model_a:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    system_prompt: "You are an academic literature recommendation expert. Recommend {num_refs} real papers in BibTeX format. Only recommend papers you are confident actually exist."

  model_b:
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "${DASHSCOPE_API_KEY}"
    model: "qwen3-max"
    system_prompt: "You are an academic literature recommendation expert. Recommend {num_refs} real papers in BibTeX format. Only recommend papers you are confident actually exist."
```

## Full config reference

### task

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | Evaluation task description |
| `scenario` | No | Usage scenario |

### dataset

| Field | Default | Description |
|-------|---------|-------------|
| `path` | — | Path to JSON/JSONL dataset file (required) |
| `shuffle` | `false` | Shuffle queries before evaluation |
| `max_queries` | `null` | Max queries to use (`null` = all) |

### target_endpoints.\<name\>

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | — | API base URL (required) |
| `api_key` | — | API key, supports `${ENV_VAR}` (required) |
| `model` | — | Model name (required) |
| `system_prompt` | built-in | System prompt; use `{num_refs}` placeholder |
| `max_concurrency` | `5` | Max concurrent requests for this endpoint |
| `extra_params` | — | Extra API request params (e.g. `temperature`) |
| `tool_config.enabled` | `false` | Enable ReAct agent with Tavily web search |
| `tool_config.tavily_api_key` | env var | Tavily API key |
| `tool_config.max_iterations` | `10` | Max ReAct iterations (1–30) |
| `tool_config.search_depth` | `"advanced"` | `"basic"` or `"advanced"` |

### verification

| Field | Default | Description |
|-------|---------|-------------|
| `crossref_mailto` | — | Email for Crossref polite pool |
| `pubmed_api_key` | — | PubMed API key |
| `max_workers` | `10` | Concurrent verification threads (1–50) |
| `timeout` | `30` | Per-request timeout in seconds |
| `verified_threshold` | `0.7` | Min composite score to count as VERIFIED |

### evaluation

| Field | Default | Description |
|-------|---------|-------------|
| `timeout` | `120` | Model API request timeout in seconds |
| `retry_times` | `3` | Number of retry attempts |

### output

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `./evaluation_results/ref_hallucination_arena` | Output directory |
| `save_queries` | `true` | Save loaded queries |
| `save_responses` | `true` | Save model responses |
| `save_details` | `true` | Save verification details |

### report

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable report generation |
| `language` | `"zh"` | Report language: `"zh"` or `"en"` |
| `include_examples` | `3` | Examples per section (1–10) |
| `chart.enabled` | `true` | Generate charts |
| `chart.orientation` | `"vertical"` | `"horizontal"` or `"vertical"` |
| `chart.show_values` | `true` | Show values on bars |
| `chart.highlight_best` | `true` | Highlight best model |

## Dataset format

Each query in the JSON/JSONL dataset:

```json
{
  "query": "Please recommend papers on Transformer architectures for NLP.",
  "discipline": "computer_science",
  "num_refs": 5,
  "language": "en",
  "year_constraint": {"min_year": 2020}
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `query` | Yes | Prompt for reference recommendation |
| `discipline` | No | `computer_science`, `biomedical`, `physics`, `chemistry`, `social_science`, `interdisciplinary`, `other` |
| `num_refs` | No | Expected number of references (default: 5) |
| `language` | No | `"zh"` or `"en"` (default: `"zh"`) |
| `year_constraint` | No | `{"exact": 2023}`, `{"min_year": 2020}`, `{"max_year": 2015}`, or `{"min_year": 2020, "max_year": 2024}` |

Official dataset: [OpenJudge/ref-hallucination-arena](https://huggingface.co/datasets/OpenJudge/ref-hallucination-arena)

## Interpreting results

**Overall accuracy (verification rate):**
- **> 75%** — Excellent: model rarely hallucinates references
- **60–75%** — Good: most references are real, some fabrication
- **40–60%** — Fair: significant hallucination, use with caution
- **< 40%** — Poor: model frequently fabricates references

**Per-field accuracy:**
- `title_accuracy` — % of titles matching real papers
- `author_accuracy` — % of correct author lists
- `year_accuracy` — % of correct publication years
- `doi_accuracy` — % of valid DOIs

**Verification status:**
- `VERIFIED` — title + author + year all exactly match a real paper
- `SUSPECT` — partial match (e.g. title matches but authors differ)
- `NOT_FOUND` — no match in any database
- `ERROR` — API timeout or network failure

**Ranking order:** overall accuracy → year compliance rate → avg confidence → completeness

## Output files

```
evaluation_results/ref_hallucination_arena/
├── evaluation_report.md          # Detailed Markdown report
├── evaluation_results.json       # Rankings, per-field accuracy, scores
├── verification_chart.png        # Per-field accuracy bar chart
├── discipline_chart.png          # Per-discipline accuracy chart
├── queries.json                  # Loaded evaluation queries
├── responses.json                # Raw model responses
├── extracted_refs.json           # Extracted BibTeX references
├── verification_results.json     # Per-reference verification details
└── checkpoint.json               # Pipeline checkpoint for resume
```

## API key by model

| Model prefix | Environment variable |
|-------------|---------------------|
| `gpt-*`, `o1-*`, `o3-*` | `OPENAI_API_KEY` |
| `claude-*` | `ANTHROPIC_API_KEY` |
| `qwen-*`, `dashscope/*` | `DASHSCOPE_API_KEY` |
| `deepseek-*` | `DEEPSEEK_API_KEY` |
| Custom endpoint | set `api_key` + `base_url` in config |

## Additional resources

- Full config examples: [cookbooks/ref_hallucination_arena/examples/](../../cookbooks/ref_hallucination_arena/examples/)
- Documentation: [docs/validating_graders/ref_hallucination_arena.md](../../docs/validating_graders/ref_hallucination_arena.md)
- Official dataset: [HuggingFace](https://huggingface.co/datasets/OpenJudge/ref-hallucination-arena)
- Leaderboard: [openjudge.me/leaderboard](https://openjudge.me/leaderboard)
