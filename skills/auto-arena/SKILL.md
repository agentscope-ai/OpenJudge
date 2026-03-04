---
name: auto-arena
description: >
  Automatically evaluate and compare multiple AI models or agents without
  pre-existing test data. Generates test queries from a task description,
  collects responses from all target endpoints, auto-generates evaluation
  rubrics, runs pairwise comparisons via a judge model, and produces
  win-rate rankings with reports and charts. Supports checkpoint resume,
  incremental endpoint addition, and judge model hot-swap.
  Use when the user asks to compare, benchmark, or rank multiple models
  or agents on a custom task, or run an arena-style evaluation.
---

# Auto Arena Skill

End-to-end automated model comparison using the OpenJudge `AutoArenaPipeline`:

1. **Generate queries** — LLM creates diverse test queries from task description
2. **Collect responses** — query all target endpoints concurrently
3. **Generate rubrics** — LLM produces evaluation criteria from task + sample queries
4. **Pairwise evaluation** — judge model compares every model pair (with position-bias swap)
5. **Analyze & rank** — compute win rates, win matrix, and rankings
6. **Report & charts** — Markdown report + win-rate bar chart + optional matrix heatmap

## Prerequisites

```bash
# Install OpenJudge
pip install py-openjudge

# Extra dependency for auto_arena (chart generation)
pip install matplotlib
```

## Gather from user before running

| Info | Required? | Notes |
|------|-----------|-------|
| Task description | Yes | What the models/agents should do (set in config YAML) |
| Target endpoints | Yes | At least 2 OpenAI-compatible endpoints to compare |
| Judge endpoint | Yes | Strong model for pairwise evaluation (e.g. `gpt-4`, `qwen-max`) |
| API keys | Yes | Env vars: `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, etc. |
| Number of queries | No | Default: `20` |
| Seed queries | No | Example queries to guide generation style |
| System prompts | No | Per-endpoint system prompts |
| Output directory | No | Default: `./evaluation_results` |
| Report language | No | `"zh"` (default) or `"en"` |

## Quick start

### CLI

```bash
# Run evaluation
python -m cookbooks.auto_arena --config config.yaml --save

# Use pre-generated queries
python -m cookbooks.auto_arena --config config.yaml \
  --queries_file queries.json --save

# Start fresh, ignore checkpoint
python -m cookbooks.auto_arena --config config.yaml --fresh --save

# Re-run only pairwise evaluation with new judge model
# (keeps queries, responses, and rubrics)
python -m cookbooks.auto_arena --config config.yaml --rerun-judge --save
```

### Python API

```python
import asyncio
from cookbooks.auto_arena.auto_arena_pipeline import AutoArenaPipeline

async def main():
    pipeline = AutoArenaPipeline.from_config("config.yaml")
    result = await pipeline.evaluate()

    print(f"Best model: {result.best_pipeline}")
    for rank, (model, win_rate) in enumerate(result.rankings, 1):
        print(f"{rank}. {model}: {win_rate:.1%}")

asyncio.run(main())
```

### Minimal Python API (no config file)

```python
import asyncio
from cookbooks.auto_arena.auto_arena_pipeline import AutoArenaPipeline
from cookbooks.auto_arena.schema import OpenAIEndpoint

async def main():
    pipeline = AutoArenaPipeline(
        task_description="Customer service chatbot for e-commerce",
        target_endpoints={
            "gpt4": OpenAIEndpoint(
                base_url="https://api.openai.com/v1",
                api_key="sk-...",
                model="gpt-4",
            ),
            "qwen": OpenAIEndpoint(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-...",
                model="qwen-max",
            ),
        },
        judge_endpoint=OpenAIEndpoint(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4",
        ),
        num_queries=20,
    )
    result = await pipeline.evaluate()
    print(f"Best: {result.best_pipeline}")

asyncio.run(main())
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML configuration file (required) |
| `--output_dir` | config value | Override output directory |
| `--queries_file` | — | Path to pre-generated queries JSON (skip generation) |
| `--save` | `False` | Save results to file |
| `--fresh` | `False` | Start fresh, ignore checkpoint |
| `--rerun-judge` | `False` | Re-run pairwise evaluation only (keep queries/responses/rubrics) |

## Minimal config file

```yaml
task:
  description: "Academic GPT assistant for research and writing tasks"

target_endpoints:
  model_v1:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
  model_v2:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"

judge_endpoint:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
```

## Full config reference

### task

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | Clear description of the task models will be tested on |
| `scenario` | No | Usage scenario for additional context |

### target_endpoints.\<name\>

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | — | API base URL (required) |
| `api_key` | — | API key, supports `${ENV_VAR}` (required) |
| `model` | — | Model name (required) |
| `system_prompt` | — | System prompt for this endpoint |
| `extra_params` | — | Extra API params (e.g. `temperature`, `max_tokens`) |

### judge_endpoint

Same fields as `target_endpoints.<name>`. Use a strong model (e.g. `gpt-4`, `qwen-max`) with low temperature (~0.1) for consistent judgments.

### query_generation

| Field | Default | Description |
|-------|---------|-------------|
| `num_queries` | `20` | Total number of queries to generate |
| `seed_queries` | — | Example queries to guide generation |
| `categories` | — | Query categories with weights for stratified generation |
| `endpoint` | judge endpoint | Custom endpoint for query generation |
| `queries_per_call` | `10` | Queries generated per API call (1–50) |
| `num_parallel_batches` | `3` | Parallel generation batches |
| `temperature` | `0.9` | Sampling temperature (0.0–2.0) |
| `top_p` | `0.95` | Top-p sampling (0.0–1.0) |
| `max_similarity` | `0.85` | Dedup similarity threshold (0.0–1.0) |
| `enable_evolution` | `false` | Enable Evol-Instruct complexity evolution |
| `evolution_rounds` | `1` | Evolution rounds (0–3) |
| `complexity_levels` | `["constraints", "reasoning", "edge_cases"]` | Evolution strategies |

### evaluation

| Field | Default | Description |
|-------|---------|-------------|
| `max_concurrency` | `10` | Max concurrent API requests |
| `timeout` | `60` | Request timeout in seconds |
| `retry_times` | `3` | Retry attempts for failed requests |

### output

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `./evaluation_results` | Output directory |
| `save_queries` | `true` | Save generated queries |
| `save_responses` | `true` | Save model responses |
| `save_details` | `true` | Save detailed results |

### report

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable Markdown report generation |
| `language` | `"zh"` | Report language: `"zh"` or `"en"` |
| `include_examples` | `3` | Examples per section (1–10) |
| `chart.enabled` | `true` | Generate win-rate chart |
| `chart.orientation` | `"horizontal"` | `"horizontal"` or `"vertical"` |
| `chart.show_values` | `true` | Show values on bars |
| `chart.highlight_best` | `true` | Highlight best model |
| `chart.matrix_enabled` | `false` | Generate win-rate matrix heatmap |
| `chart.format` | `"png"` | Chart format: `"png"`, `"svg"`, or `"pdf"` |

## Interpreting results

**Win rate:** percentage of pairwise comparisons a model wins. Each pair is evaluated in both orders (original + swapped) to eliminate position bias.

**Rankings example:**
```
  1. gpt4_baseline       [################----] 80.0%
  2. qwen_candidate      [############--------] 60.0%
  3. llama_finetuned      [##########----------] 50.0%
```

**Win matrix:** `win_matrix[A][B]` = how often model A beats model B across all queries.

## Checkpoint & resume

The pipeline saves progress after each step. Interrupted runs resume automatically:

- `--fresh` — ignore checkpoint, start from scratch
- `--rerun-judge` — re-run only the pairwise evaluation step (useful when switching judge models); keeps queries, responses, and rubrics intact
- Adding new endpoints to config triggers incremental response collection; existing responses are preserved

## Output files

```
evaluation_results/
├── evaluation_results.json     # Rankings, win rates, win matrix
├── evaluation_report.md        # Detailed Markdown report (if enabled)
├── win_rate_chart.png          # Win-rate bar chart (if enabled)
├── win_rate_matrix.png         # Matrix heatmap (if matrix_enabled)
├── queries.json                # Generated test queries
├── responses.json              # All model responses
├── rubrics.json                # Generated evaluation rubrics
├── comparison_details.json     # Pairwise comparison details
└── checkpoint.json             # Pipeline checkpoint
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

- Full config examples: [cookbooks/auto_arena/examples/](../../cookbooks/auto_arena/examples/)
- Documentation: [Auto Arena Guide](https://agentscope-ai.github.io/OpenJudge/applications/auto_arena/)
