# Eval Pipeline Functional Testing Guide

This guide explains how to prove the eval-pipeline skills work on realistic scenarios.

## What To Test

There are two different test types:

1. Skill package quality test
   - Use `cookbooks/skills_evaluation/evaluate_skills.py`.
   - This grades whether the SKILL.md files are complete, relevant, safe, and well-designed.

2. Functional scenario test
   - Use `skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py`.
   - This checks whether a skill actually guides an agent to the expected evaluation workflow for a realistic user request.

For this project, the second test is more important. It demonstrates that the methodology helps users build an evaluation system.

## API Key

To run the functional tests you need an OpenAI-compatible API key. Two roles are used:

- **actor**: follows the skill and answers the test request (default `qwen3.6-plus`)
- **judge**: grades the actor output against acceptance criteria (default `qwen3-max`)

The actor and judge default to *different* models on purpose — one model judging its own
output inflates pass rates.

### Option A — Aliyun DashScope (Bailian)

Put `DASHSCOPE_API_KEY` in `.env` at the repo root. The runner detects it and uses the
compatible-mode endpoint automatically — no other config needed:

```bash
# .env
DASHSCOPE_API_KEY=sk-...
```

The endpoint is `https://dashscope.aliyuncs.com/compatible-mode/v1` (note: `aliyuncs.com`,
not `aliyun.com`). Verified models: `qwen3.6-plus`, `qwen-plus`, `qwen-max`, `qwen3-max`,
`qwen-flash`, `qwen-turbo`.

### Option B — OpenAI or any OpenAI-compatible provider

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-provider.example/v1"   # omit for OpenAI itself
export OPENAI_MODEL="qwen3.6-plus"        # actor, optional
export OPENAI_JUDGE_MODEL="qwen3-max"     # judge, optional
```

You can also override per run: `--model <actor>` and `--judge-model <judge>`.

The runner retries transient 429s (per-minute token buckets) with exponential backoff,
so a burst limit won't fail a run.

## Quick Smoke Test

Run one simple routing case first:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --case-id meta_eval_001_route_from_nothing
```

Run three representative cases:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --case-id eval_design_001_from_trace_failures \
  --case-id metric_design_001_deterministic_first \
  --case-id align_human_002_insufficient_labels
```

Run all cases:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py
```

Outputs:

```text
skills/eval_pipeline/tests/results/eval_pipeline_functional_report.md
skills/eval_pipeline/tests/results/eval_pipeline_functional_results.json
```

The repo ignores `*.json`, so use the Markdown report for review and versioned summaries.

## Recommended Test Samples

Start with this minimal set:

| Purpose | Case ID | Why |
|---|---|---|
| Router correctness | `meta_eval_001_route_from_nothing` | Ensures no-data users go to bootstrap. |
| Dataset design | `eval_design_001_from_trace_failures` | Ensures trace failures become dimensions and strata. |
| Metric design | `metric_design_001_deterministic_first` | Ensures deterministic graders are preferred when possible. |
| Human calibration | `align_human_002_insufficient_labels` | Ensures the skill refuses premature auto-gating. |
| RAG scenario | `rag_eval_001_diagnose_generation_problem` | Ensures retrieval and generation are separated. |
| Prompt regression | `prompt_regression_002_too_few_samples` | Ensures no winner is declared from too few samples. |
| Safety | `redteam_001_policy_first` | Ensures attacks are policy-derived and paired with over-refusal. |
| End-to-end | `end_to_end_001_support_eval_system` | Ensures the collection orders workflows correctly. |

This gives good signal without paying to run all 18 cases.

## Handling variance (run more than once)

Both the actor and the judge are LLMs, so a single run is noisy — the same unchanged skill
can score `pass` on one run and `partial` on the next. Do not conclude a skill is fixed (or
broken) from one run near the boundary.

Use `--repeat N` to run each case N times and report the majority verdict + mean score
(per-run spread is kept in the report):

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --case-id end_to_end_001_support_eval_system --repeat 3
```

Rule of thumb: `--repeat 1` for a quick scan, `--repeat 3` (or 5) before claiming a skill
change worked.

## How To Interpret Results

Treat this as a forward test, not a mathematical proof.

- `pass`: the skill gave enough guidance to satisfy all critical acceptance criteria.
- `partial`: the skill mostly worked but missed an artifact, caveat, or threshold.
- `fail`: the skill routed incorrectly, violated a hard gate, or produced a misleading workflow.

Useful pass threshold for the first audit:

- smoke set: 8/8 pass or partial, with at least 6 pass
- full set: at least 80% pass, no failures on hard-gate cases

Hard-gate cases:

- `align_human_002_insufficient_labels`
- `prompt_regression_002_too_few_samples`
- `redteam_002_regulated_signoff`
- `bootstrap_001_cold_start`

## What To Do After A Failure

1. Read the actor output in the Markdown report.
2. Compare it to the missed acceptance criteria.
3. Patch the relevant SKILL.md with a clearer hard rule, output schema, or SDK example.
4. Re-run only that case.
5. Re-run the smoke set before accepting the change.
