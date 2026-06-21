---
name: eval-report
description: >
  Use when the user has run multiple evaluation skills and wants a comprehensive
  analysis — maturity assessment, cross-skill signals, trends, prioritized actions,
  and an executive summary. Also use when the user mentions eval health check,
  evaluation audit, ship readiness, evaluation maturity, or "how good is my
  evaluation system itself." This is a read-only analysis skill.
---

<HARD-GATE>
NO recommendation WITHOUT statistical evidence backing it.
NO "system ready" declaration WITHOUT all calibrated judges passing AND all production gates green.
NO trend analysis WITHOUT at least 2 data points in history.
</HARD-GATE>

# Eval Report

Synthesize everything from your evaluation journey into a comprehensive report.
This skill is read-only — it analyzes what exists, doesn't create new graders or datasets.

## When to Activate

- You've run 2+ evaluation skills and want the big picture
- You need to report evaluation status to non-technical stakeholders
- You're making a ship/no-ship decision and need evidence
- The evaluation system has been running for a while — time for a health check

## Checklist

You MUST create a task for each item and complete them in order:

1. **Inventory scan** — catalog everything in eval-design.md + runs/ history
2. **Maturity assessment** — 5 dimensions × 4 levels
3. **Cross-skill signal synthesis** — consistent findings + contradictions
4. **Weakness diagnosis** — failure concentration, correlations, stratum gaps
5. **Root cause classification** — system / metric / data / unclear
6. **Prioritized recommendations** — P0/P1/P2 actions with impact estimates
7. **Executive summary** — ship readiness + top 3 risks + next actions

## Step 1: Inventory Scan

Read `eval-design.md` and all `runs/` directories. Build a timeline:

```
Timeline:
  2026-04-15  01-eval-design   → 5 failure modes → 3 dimensions from 200 traces
  2026-04-18  02-metric-design → 4 graders configured (2 LLM + 1 rule + 1 executable)
  2026-04-25  (evaluation run) → 90-sample stratified dataset scored
  2026-05-01  03-align-human   → 2 judges Phase 3, 1 Phase 2, 1 Phase 1 (TPR/TNR + kappa)
  2026-05-10  07-redteam       → safety audit not yet run
```

Report key metrics:
- Total skills run, total principles, total labels
- Calibrated judges: X of Y (with TPR/TNR range)
- Last activity date per skill

## Step 2: Maturity Assessment

Rate the evaluation system across 5 dimensions:

| Dimension | L1 (Initial) | L2 (Developing) | L3 (Established) | L4 (Optimizing) |
|-----------|-------------|-----------------|-------------------|-----------------|
| **Failure Discovery** | No systematic analysis | Failure modes identified | Coverage validated with stratification | Continuous triage from production |
| **Judge Quality** | v0 uncalibrated only | Some calibrated (TPR/TNR measured) | All calibrated with CI | Calibrated + aligned with humans |
| **Label Coverage** | < 50 labels | 50-200 labels | 200+ stratified labels | Coverage audit passed, drift monitored |
| **Safety Coverage** | No redteaming | Ad-hoc redteam run | Systematic redteam with policy doc | Continuous redteam with sign-off |
| **Human Alignment** | No alignment data | Kappa measured for some judges | Kappa ≥ 0.8 for all judges | Human spot-check only, quarterly audit |

**Scoring rule**: The overall maturity level is the **minimum** across dimensions
(weakest link principle). If 4 dimensions are L3 but Safety is L1, the system is L1.

## Step 3: Cross-Skill Signal Synthesis

### Consistent Signals (high confidence)

Find themes confirmed by multiple skills. Example:
- "Factuality is the top risk" — evidence chain:
  - 01-eval-design: #1 failure mode (38% prevalence in traces)
  - 02-metric-design: weighted as the highest-impact dimension
  - 03-align-human: TPR=0.92 TNR=0.88 (confirmed measurable)

### Contradictions (needs investigation)

Find where skills disagree. These are the most valuable findings:
- "02-metric-design weighted hallucination as a top signal, but 03-align-human shows
  the hallucination judge has TPR=0.74"
  → Possible explanations: the judge prompt captures surface patterns, not real
    hallucination. Or the judge prompt needs refinement, or the labels are noisy.

### Coverage Gaps (blind spots)

What hasn't been touched by any skill?
- "01-eval-design coverage shows multilingual input_type n=0, no workflow has addressed
  non-English queries"

## Step 4: Weakness Diagnosis

### Failure Concentration

Which principle/grader has the lowest pass rate? Where are failures clustering?

### Failure Correlation

Compute Jaccard similarity between principle pairs — when sample A fails on principle X,
does it also fail on principle Y? Highly correlated pairs (Jaccard > 0.5) likely share
a root cause.

### Per-Stratum Weakness

Which difficulty stratum performs worst across all principles? If boundary stratum
TPR < 0.7 for 3 of 4 principles, boundary discrimination is a systemic weakness.

## Step 5: Root Cause Classification

For each weakness area, classify the root cause:

| Type | Definition | Key indicator |
|------|-----------|---------------|
| **system_problem** | The application itself performs poorly | Low pass rate + high judge-human agreement |
| **metric_problem** | The judge/eval is flawed | Low pass rate + low judge-human agreement |
| **data_problem** | The eval dataset isn't representative | 01-eval-design coverage shows thin strata OR label drift detected |
| **unclear** | Not enough evidence | Conflicting signals, need more data |

This classification is critical — fixing a metric problem by changing the system
(or vice versa) wastes effort.

## Step 6: Prioritized Recommendations

Generate P0/P1/P2 actions. Each must include: priority, concrete action, current
state, target state, expected impact, and the skill to use.

```
🔴 P0 | Calibrate hallucination judge
       Current: TPR=0.74 (below 0.8 threshold)
       Target:  TPR >= 0.8
       Impact:  Judge becomes usable as production gate
       Use:     03-align-human

🔴 P0 | Add boundary samples for hallucination
       Current: n=6 boundary samples (CI half-width ±18%)
       Target:  n >= 20 (CI narrows to ±10%)
       Impact:  Reliable per-stratum TPR measurement
       Use:     01-eval-design

🟡 P1 | Align tone_consistency judge
       Current: kappa=0.72, bias=-0.15 (lenient)
       Target:  kappa >= 0.8, |bias| < 0.1
       Impact:  Reduce false positive rate ~15%
       Use:     03-align-human

🟢 P2 | Enable auto-gate for factuality judge
       Current: kappa=0.87, TPR=0.92, TNR=0.88
       Impact:  Eliminate 90% of human review for this dimension
       Use:     03-align-human (mark Phase 3)
```

## Step 7: Executive Summary

One page for non-technical stakeholders:

```
Executive Summary
=================
System: Customer support chatbot for e-commerce
Stakes: Production
Report Date: 2026-05-12

SHIP READINESS: Conditional
  2 items must be resolved before production gate:
  1. Hallucination judge TPR below threshold (0.74 < 0.8)
  2. No safety/redteam evaluation has been run

TOP 3 RISKS:
  1. Hallucination detection unreliable — severity: HIGH
     The judge measuring whether the bot fabricates information itself has poor
     recall (TPR=0.74), meaning ~26% of hallucinations go undetected.
     Mitigation: Calibrate with more boundary labels (2-3 weeks).

  2. Safety coverage missing — severity: MEDIUM
     No jailbreak, injection, or PII leakage testing has been performed.
     Mitigation: Run redteam skill this sprint (1-2 days).

  3. Tone evaluation is biased lenient — severity: LOW
     The tone judge systematically rates responses as better than humans do.
     Mitigation: Refine judge prompt with borderline examples.

EVAL MATURITY: L2 (Developing) → Target L3 in 3-4 weeks
  Strongest: Failure Discovery (L3)
  Weakest:  Safety Coverage (L1), Human Alignment (L2)

NEXT ACTIONS (this sprint):
  [P0] Add boundary labels + recalibrate hallucination judge
  [P0] Run initial redteam evaluation
  [P1] Refine tone judge alignment
```

## Output

| File | Content |
|------|---------|
| `eval-design.md` | `eval_report:` namespace (maturity, signals, trends, actions) |
| `runs/eval-report/<ts>/report.md` | Full analysis report |
| `runs/eval-report/<ts>/executive-summary.md` | One-page stakeholder summary |

Each per-metric run you synthesize should be a `runs/<skill>/<ts>/results.json` row of the form:

```python
{"metric": "order_accuracy", "mean": 0.82, "ci_95": [0.78, 0.86], "n": 90,
 "by_stratum": {"easy": 0.95, "boundary": 0.71, "adversarial": 0.60},
 "verdict": "pass | fail | insufficient_evidence"}
```

## Common Mistakes

- **Giving recommendations without root cause classification.** "Pass rate is low"
  doesn't tell you what to fix. Classify as system/metric/data problem first.
- **Maturity scoring by gut feel.** Each L1-L4 rating must cite specific evidence
  from eval-design.md fields.
- **Executive summary too long.** It has one audience: busy stakeholders. Three
  risks, three actions, one page. Details go in the full report.
- **Ignoring cross-skill contradictions.** When two skills disagree about the same
  dimension, that's the most important signal in the report — it reveals a structural
  issue in assumptions or methodology.
- **All recommendations marked P1.** If everything is medium priority, nothing is.
  P0 = blocks ship. P1 = important this sprint. P2 = backlog.

## Next Skills

After `04-eval-report`:
- Recommendations point to specific workflows (03-align-human, 01-eval-design, 07-redteam, etc.)
  based on identified weaknesses. This skill is the "end of loop" analysis —
  after addressing recommendations, run this skill again to track progress.