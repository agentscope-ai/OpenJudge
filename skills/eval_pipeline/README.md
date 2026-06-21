# Eval Pipeline — a methodology for building your own evaluation system

A set of skills that walk an evaluation practitioner through building a scenario-specific
evaluation system on top of the OpenJudge SDK: define the eval set, design metrics/graders,
decide whether and how to bring humans in for validation, and handle vertical scenarios
(RAG, prompt A/B, safety, cold start).

Each skill is a self-contained workflow in `<NN-name>/SKILL.md`. Start at `00-meta-eval`,
which diagnoses what you have and routes you to the right one.

## The workflows

| # | Skill | Use it when |
|---|---|---|
| 00 | `00-meta-eval` | You don't know where to start. It routes you (and holds the canonical lifecycle below). |
| 01 | `01-eval-design` | You have traces or a spec and need a stratified, OpenJudge-ready dataset. |
| 02 | `02-metric-design` | You need to pick graders and build a runnable `GradingRunner` pipeline (incl. hard gates). |
| 03 | `03-align-human` | You have ≥50 human labels and need to know if the judge is trustworthy (TPR/TNR, kappa, bias). |
| 04 | `04-eval-report` | You've run several workflows and want maturity + ship-readiness synthesis. |
| 05 | `05-rag-eval` | RAG system — separate retrieval vs generation failure. |
| 06 | `06-prompt-regression` | You changed a prompt and need a position-debiased, CI-backed A/B verdict. |
| 07 | `07-redteam` | Safety/security audit — policy-derived attacks paired with over-refusal. |
| 08 | `08-bootstrap` | You have nothing — get an uncalibrated v0 + a roadmap to calibration. |

## Canonical lifecycle

```
00-meta-eval → (01-eval-design | 08-bootstrap) → 02-metric-design → run →
   03-align-human (once ≥50 labels; required before any production gate) →
   scenario module (05 / 06 / 07 as needed) → 04-eval-report
```

Preconditions worth remembering: `06-prompt-regression` needs paired baseline+candidate
outputs on shared queries; nothing is "production-ready" without labels + `03-align-human`.

## Bundled scripts (run, don't re-implement)

Five skills ship a **standalone, tested** analysis script in their own `scripts/` folder.
They use the Python standard library only (numpy optional) and have **no OpenJudge
dependency** — they read JSON/JSONL and print a report, so an agent runs them instead of
re-deriving fragile statistics. Each has `--self-test`.

| Skill | Script | Computes |
|---|---|---|
| 01 | `scripts/coverage_check.py` | per-dimension / per-stratum coverage, thin-cell flags |
| 03 | `scripts/calibration.py` | TPR/TNR/F1, kappa, Gwet's AC1, bias, bootstrap CI, calibration gate |
| 05 | `scripts/rag_diagnostic.py` | retrieval-vs-generation 2×2 diagnostic matrix |
| 06 | `scripts/pairwise.py` | position-debiased win-rate + bootstrap CI verdict |
| 07 | `scripts/asr_report.py` | ASR per category/vector paired with over-refusal rate |

## OpenJudge is optional

The **methodology and the analysis scripts need no OpenJudge install** — you can apply the
whole pipeline with your own judge and just feed the scripts JSON/JSONL. OpenJudge is used
only to *produce* those inputs (running graders) and is required by two SDK-centric skills:
`02-metric-design` (grader selection / `GradingRunner` / aggregators) and `08-bootstrap`
(rubric generators). Those skills state the dependency up front.

## Self-contained skills

Each `<NN-name>/SKILL.md` is **self-contained** — it carries inline the data shapes,
statistics, and data principles it needs, so any single skill folder can be installed and
used on its own (per the Anthropic Agent Skill protocol). The recurring canon you will see
across skills:

- **Data shapes** — dataset row, labels row, RAG trace, pairwise row, redteam rows,
  run-result (each skill shows the shape it uses).
- **Statistics** — sample-size thresholds, bootstrap 95% CI, TPR/TNR, Cohen's kappa vs
  Gwet's AC1, position debiasing (in `03` and `06`).
- **Data principles** — never LLM-generate labels, judge≠target, stratify by difficulty,
  coverage checks, uncalibrated-until-proven (in `01`, `07`, `08`).

## Models / SDK

Graders run on any OpenJudge `BaseChatModel`. The examples use `OpenAIChatModel`, which reads
`OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment. For Aliyun DashScope (Bailian):

```bash
OPENAI_API_KEY=<dashscope key>
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # note: aliyuncs.com
```

## Validating the skills themselves

These skills are tested with a **forward functional harness**: an *actor* model follows a skill
to answer a realistic user request, and a separate, stronger *judge* model grades that answer
against the case's acceptance criteria. Actor and judge default to different models on purpose —
one model judging its own output inflates pass rates.

Everything needed to reproduce the suite lives in `tests/`:

| File | Role |
|---|---|
| `tests/run_eval_pipeline_skill_tests.py` | The runner (paths are derived from its own location, so it works from anywhere). |
| `tests/eval_pipeline_test_cases.jsonl` | The 18 test cases (kept tracked via a `.gitignore` exception). |
| `tests/eval_pipeline_testing_guide.md` | Full guide: smoke set, recommended samples, how to read results. |
| `tests/eval_pipeline_skill_audit.md` | Findings from prior audit runs. |

### 1. Prerequisites

```bash
pip install -e ".[dev]"          # brings in openai + python-dotenv (already project deps)
```

### 2. API key

Two model roles need an OpenAI-compatible key. The runner auto-loads `.env` from the **repo
root**. Either option works:

```bash
# Option A — Aliyun DashScope (Bailian): put this in <repo-root>/.env, nothing else needed.
DASHSCOPE_API_KEY=sk-...
# Endpoint defaults to https://dashscope.aliyuncs.com/compatible-mode/v1 (note: aliyuncs.com)

# Option B — OpenAI or any compatible provider:
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-provider.example/v1"   # omit for OpenAI itself
```

Defaults: actor `qwen3.6-plus`, judge `qwen3-max`. Override per run with `--model` /
`--judge-model`, or via `OPENAI_MODEL` / `OPENAI_JUDGE_MODEL`.

### 3. Run

```bash
# Smoke — one routing case (fast sanity check that the harness works end to end):
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --case-id meta_eval_001_route_from_nothing

# A representative smoke set:
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --case-id eval_design_001_from_trace_failures \
  --case-id metric_design_001_deterministic_first \
  --case-id align_human_002_insufficient_labels

# Full suite, 3 repeats per case (recommended before claiming a skill change worked):
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py --repeat 3
```

Actor and judge are both LLMs, so a single run is noisy. Use `--repeat 3` (or 5) and read the
majority verdict + mean score before drawing conclusions.

### 4. Output

Results are written to `tests/results/` (git-ignored — they regenerate on every run):

```text
tests/results/eval_pipeline_functional_report.md     # human-readable table + per-case detail
tests/results/eval_pipeline_functional_results.json  # raw scores
```

Verdicts: `pass` (all critical criteria met) · `partial` (mostly worked, missed a
caveat/threshold) · `fail` (routed wrong, broke a hard gate, or gave a misleading workflow).
See the testing guide for the full pass thresholds and the list of hard-gate cases.
