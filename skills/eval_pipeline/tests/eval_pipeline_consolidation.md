# Eval Pipeline Consolidation Plan

Date: 2026-06-19

> **Superseded note (2026-06-21):** the "create a shared `references/` directory" parts of
> this plan were reversed. Skills must be *independently installable* (Anthropic Agent Skill
> protocol), so shared content lives **inline in each skill**, not in a collection-level
> `references/`. The merge/keep analysis below still stands.

This document focuses on which eval-pipeline skills should stay separate, merge, or be demoted to shared references.

## Recommended Shape

Keep the collection, but reorganize it around three layers:

1. Core methodology workflows
   - `00-meta-eval`
   - `01-eval-design`
   - `02-metric-design`
   - `03-align-human`
   - `04-eval-report`

2. Scenario workflows
   - `05-rag-eval`
   - `06-prompt-regression`
   - `07-redteam`

3. Fast-start workflow
   - `08-bootstrap`

This is already close to the current structure. The useful change is not a big deletion. It is removing repeated explanations and making shared contracts explicit.

## Merge Or Keep

| Current Skill | Recommendation | Reason |
|---|---|---|
| `00-meta-eval` | Keep, rename conceptually to router/orchestrator | It is the entry point and should explain the canonical path. |
| `01-eval-design` | Keep | Dataset design is a distinct practitioner task. |
| `02-metric-design` | Keep, but split references if it grows | Metric/grader design is the most OpenJudge-specific workflow. |
| `03-align-human` | Keep | Human calibration is a distinct trust/validation stage. |
| `04-eval-report` | Keep | It is the end-of-loop synthesis and maturity audit. |
| `05-rag-eval` | Keep as scenario skill | RAG has a distinct retrieval/generation diagnostic matrix. |
| `06-prompt-regression` | Keep as scenario skill | Prompt A/B regression has distinct paired-comparison and CI logic. |
| `07-redteam` | Keep as scenario skill | Safety testing has policy, ASR, and over-refusal requirements. |
| `08-bootstrap` | Keep, but position as fast-start wrapper | It overlaps with `01` and `02`, but is valuable for no-data users. |

## Redundancy To Remove

### 1. Repeated "do not generate labels" rule

Appears in:

- `01-eval-design`
- `07-redteam`
- `08-bootstrap`

Keep it in each skill as a short hard rule, but move the longer rationale into a shared reference such as:

`skills/eval_pipeline/references/eval-data-principles.md`

### 2. Repeated sample-size and confidence guidance

Appears in:

- `01-eval-design`
- `03-align-human`
- `06-prompt-regression`

Keep the threshold in each skill, but centralize formulas and examples in:

`skills/eval_pipeline/references/statistics.md`

### 3. Repeated OpenJudge dataset fields

Appears in:

- `01-eval-design`
- `05-rag-eval`
- `06-prompt-regression`
- `07-redteam`
- `08-bootstrap`

Create one canonical schema reference:

`skills/eval_pipeline/references/artifact-schemas.md`

Minimum shared schemas:

- evaluation dataset row
- labels row
- metric plan
- run result
- report summary

### 4. Repeated next-step chains

Every skill has "Next Skills". This is useful, but can become inconsistent.

Recommendation:

- Keep one short "Next workflow" section per skill.
- Put the full canonical lifecycle only in `00-meta-eval`.

## Do Not Merge These

Do not merge `01-eval-design` and `02-metric-design`.

Reason: users often have one but not the other. Dataset design and metric design require different inputs and produce different artifacts.

Do not merge `03-align-human` into `02-metric-design`.

Reason: grader creation and grader validation are separate trust stages. Collapsing them encourages users to treat a newly written judge as production-ready.

Do not merge `05-rag-eval`, `06-prompt-regression`, and `07-redteam`.

Reason: these are scenario skills with different artifact shapes and success metrics.

## Bootstrap Positioning

`08-bootstrap` is the only skill with real overlap:

- It interviews like `01-eval-design`.
- It generates a grader like `02-metric-design`.
- It points to `03-align-human`.

Recommendation:

Keep it because it is the best first-run experience for users with no data. But make it explicitly a wrapper:

> Use `08-bootstrap` only when the user has no traces, labels, or existing eval set. Once artifacts exist, transition to `01-eval-design` and `02-metric-design`.

## Proposed First Revision

Do these before changing deeper content:

1. Update `00-meta-eval` wording from "Install" to "Use workflow".
2. Add a canonical lifecycle to `00-meta-eval`.
3. Add shared artifact schema references.
4. Patch SDK code examples that are likely to drift.
5. Run functional tests from `eval_pipeline_test_cases.md`.
