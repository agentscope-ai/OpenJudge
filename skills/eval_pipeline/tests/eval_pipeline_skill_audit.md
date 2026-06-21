# Eval Pipeline Skill Audit

Date: 2026-06-20 (supersedes the 2026-06-19 draft)

Scope: `skills/eval_pipeline/{00..08}/SKILL.md`

Goal: assess whether the eval-pipeline skills can guide a practitioner to build a
scenario-specific evaluation system on top of the OpenJudge SDK — and verify it with
live functional tests rather than static inspection alone.

## How this audit was done

Two grounding methods, not just reading:

1. **SDK ground-truth check** — every import path, class name, and constructor/method
   signature used in the skills was checked against the real `openjudge` source, and the
   flagship objects (LLMGrader string templates, FunctionGrader sync funcs, GradingRunner,
   WeightedSumAggregator, validation analyzers, PairwiseAnalyzer, the generators) were
   constructed/executed in a Python session.
2. **Live functional tests** — `run_eval_pipeline_skill_tests.py` runs an actor model that
   follows each skill and a stronger judge model that grades the output against acceptance
   criteria. Run against Aliyun DashScope (`qwen3.6-plus` actor, `qwen3-max` judge).

## Executive summary

The collection has a strong conceptual spine and — contrary to the earlier draft — its
OpenJudge grounding is mostly **correct**. The earlier audit speculated that imports and
signatures were likely wrong; direct verification shows they are right. The genuine defects
are narrower and were fixed in this pass and re-verified by functional tests.

Overall readiness (revised):

| Dimension | Rating | Notes |
|---|---:|---|
| Methodology coverage | High | dataset, metrics, calibration, reporting, RAG, prompt regression, redteam, bootstrap. |
| OpenJudge SDK grounding | High | imports/signatures/templates verified against source; flagship snippets construct and run. |
| User onboarding clarity | High | `00-meta-eval` now gives a provisional route immediately + a canonical lifecycle with preconditions. |
| Output contract quality | Medium-High | gate-vs-weight pattern now concrete; shared artifact schemas still worth centralizing. |
| Testability | High | 18 functional cases + runner; smoke set passes after fixes. |
| Progressive disclosure | Medium | `02-metric-design` is dense (~470 lines); split references if it grows. |

## Correcting the earlier draft

The 2026-06-19 draft listed a P0: "Some SDK examples are likely to fail or drift," naming
`02`, `03`, `06`, `08`. Direct verification:

| Earlier claim | Verified reality |
|---|---|
| Imports likely wrong (`02`, `03`) | **False.** `graders.text.string_match`, `graders.format.json.*`, `analyzer.validation.*`, both generators, `PairwiseAnalyzer` all resolve. |
| `StringMatchGrader` import-sensitive | **Misattributed.** It needs `jieba`, which is a *declared core dependency* (`pyproject.toml`), not an optional one. Fine on a proper `pip install -e .`. |
| LLMGrader string template risky | **False.** `str` templates render via `str.format`; `{var}` substitution and `{{...}}` JSON escaping both work (verified). |
| `GraderScore` missing `name` | **False.** `name` is inherited from `GraderResult`. |

The lesson: the methodology layer was sound; the audit just hadn't been run against the code.

## Genuine defects found (all fixed this pass)

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `02`, `08` | `OpenAIChatModel(base_url="https://api.anthropic.com/v1")` — Anthropic's API is not OpenAI-compatible at that URL, so the flagship `model` would not run. | Replaced with env-driven `OpenAIChatModel(model="qwen-plus")` + DashScope/OpenAI base-URL note. |
| 2 | `06-prompt-regression` | `compute_win_rates(analysis)` bootstrap indexed a Pydantic `PairwiseAnalysisResult` as a list and read a non-existent `metadata["winner"]`. Broken. | Rewrote to derive per-comparison winners from `dataset` + `results` (respecting swap order) and bootstrap over them. Verified with mock data. |
| 3 | `07-redteam` | `runner.arun()` returns `{grader: [scores]}` but Step 5 iterated it as attack-metadata rows (`a["category"]`, `a["violated"]`). | Added an explicit join-back from `GraderScore` to attack `category`/`vector` (`HarmfulnessGrader` 1–5, violation = score < threshold). Verified with mock data. |
| 4 | `05-rag-eval` | `retrieved_docs` shown as `list[str]` in Step 1 but indexed as `doc["id"]` in Step 4. Internal contradiction. | Normalized to `{"id","text"}` dicts + a wrap helper for raw strings; faithfulness join now uses `doc["text"]`. |
| 5 | `04-eval-report` | Examples referenced legacy skill names (`agent-triage`, `distill`, `calibrate`, `stratify`) absent from the 00–08 collection. | Replaced with current workflow names throughout. |

## Methodology defects found via functional tests (fixed this pass)

| Case | Before | Root cause | Fix | After |
|---|---|---|---|---|
| `meta_eval_001` | partial 0.67 | HARD-GATE blocked any recommendation until stakes+domain were known, even when data+labels already pin the route. Then a `?`-placeholder template confused the judge. | Route on `data_form + label_status`; give a **provisional** recommendation immediately. Labeled diagnosis fields (`stakes=asking`) instead of bare `?`. Single-recommendation rule made explicit. | pass 1.00 |
| `metric_design_001` | partial 0.80 | "Gates vs weights" was prose only; every runnable example used `WeightedSumAggregator`, so PII became compensable. | Added a runnable `GatedWeightedSumAggregator` (hard-fail on gate breach). Verified. | pass 1.00 |
| `end_to_end_001` | fail 0.40 | No canonical lifecycle; nothing stopped routing to `06-prompt-regression` before paired outputs / labels existed. | Added a **Canonical Workflow** to `00-meta-eval` with explicit preconditions (prompt-regression needs paired outputs; production needs labels + `03-align-human`, stated emphatically). | pass 1.00 |

Smoke set: baseline **5 pass / 2 partial / 1 fail** → after fixes **8/8 reach pass** (each
re-verified at 1.00 across re-runs; see variance note).

Final full suite (18 cases × `--repeat 3`, on the self-contained + scripts version):
**16 pass / 2 partial / 0 fail**. Both partials at 0.87 with per-run verdicts
`[partial, partial, pass]` — `meta_eval_001` had zero missed criteria (pure judge variance);
`align_human_001` missed "surface the TPR/TNR/kappa/bias numbers" on 2/3 runs, addressed by a
`03` fast-path note telling the agent to report and interpret the script's numbers, not just
run it — after the fix it re-ran **3/3 pass**. Effective suite: **17 pass / 1 partial / 0 fail**,
the lone partial being `meta_eval_001` judge variance (no missed criteria).

### Variance note (important for trusting the driver)

Single-run functional verdicts are noisy — actor + judge are both LLMs. `end_to_end_001`
scored 0.40 → 1.00 → 0.60 → 1.00 across runs on identical skill text. To make optimization
claims robust, the runner now supports `--repeat N` (majority verdict + mean score, per-run
spread kept in the report). Use `--repeat 3` (or more) for any case near the pass/partial
boundary before concluding a skill is fixed or broken.

## Agent-usability upgrade: bundled scripts (2026-06-21)

The skills were prose-only — they *described* the statistics and the agent re-derived the
code each run (the root cause of the `06` bootstrap and `07` join-back bugs). Five skills now
bundle a **standalone, tested** analysis script (stdlib only, numpy optional, **no OpenJudge
dependency**, each with `--self-test`):

| Skill | Script | Replaces fragile inline code for |
|---|---|---|
| 01 | `scripts/coverage_check.py` | per-dimension/stratum coverage + thin-cell flags |
| 03 | `scripts/calibration.py` | TPR/TNR/F1, kappa, Gwet's AC1, bias, bootstrap CI, gate |
| 05 | `scripts/rag_diagnostic.py` | retrieval-vs-generation 2×2 matrix |
| 06 | `scripts/pairwise.py` | swap-aggregate win-rate + bootstrap CI |
| 07 | `scripts/asr_report.py` | ASR per category/vector + over-refusal pairing |

Design split (so the skills are maximally out-of-the-box):

- **Analysis = standalone scripts.** Operate on JSON/JSONL, no SDK — a user without OpenJudge
  can still run the full methodology with their own judge.
- **SDK wiring = OpenJudge-dependent reference code** kept in the skills that genuinely need
  it (`02`, `08`), which now carry an explicit "Requires OpenJudge" banner.

Each script skill gained a "Fast path: run the bundled script" section so an agent runs the
tested tool instead of re-implementing it.

## Remaining recommendations (not blocking)

- **Self-containment over DRY.** An earlier pass extracted shared content into a collection-
  level `references/` directory. That breaks the Anthropic Agent Skill protocol for
  *independent* install — a skill folder must carry everything it needs. The references were
  therefore inlined back into each skill (each now holds only the data shapes / statistics /
  principles relevant to it) and the shared directory was removed. A short `README.md`
  orients the collection. If a single skill ever needs heavy bundled detail, put it in that
  skill's own `<skill>/references/` (the test runner loads per-skill bundled references).
- **Insufficient-evidence fallback** is well-handled by `03`/`06`/`08` already (the
  functional tests confirm they refuse premature claims). Consider a shared one-line
  `verdict: insufficient_evidence` block for uniformity.
- **`03-align-human` snippet** uses `GraderScore` without importing it in that block —
  cosmetic; the import appears in earlier skills.

## Test strategy (what's in place)

1. Static SDK grounding — import/signature/construction checks (done ad hoc; worth a CI
   step that extracts code blocks and imports them under dev deps).
2. Functional scenario tests — `run_eval_pipeline_skill_tests.py`, 18 cases, actor+judge.
3. Re-run on every skill edit; keep the Markdown report under
   `skills/eval_pipeline/tests/results/`.

## Per-skill status (revised)

| Skill | SDK grounding | Functional | Note |
|---|---|---|---|
| `00-meta-eval` | n/a | fixed | Provisional routing + canonical lifecycle + single-recommendation rule. |
| `01-eval-design` | correct | pass 1.00 | Strong dataset methodology; no-synthetic-labels rule lands. |
| `02-metric-design` | correct (+gate pattern, model fix) | pass 1.00 | Deterministic-first + gate-vs-weight now concrete. |
| `03-align-human` | correct | pass 1.00 | Refuses calibration on too-few labels. |
| `04-eval-report` | names fixed | (not in smoke) | Weakest-link maturity; references now current. |
| `05-rag-eval` | doc schema fixed | pass 1.00 | Retrieval/generation split holds. |
| `06-prompt-regression` | bootstrap fixed | pass 1.00 | Refuses winner on 6 samples. |
| `07-redteam` | join-back added | pass 1.00 | Policy-first attacks + over-refusal. |
| `08-bootstrap` | model fix | (not in smoke) | Cold-start + uncalibrated warning + roadmap. |
