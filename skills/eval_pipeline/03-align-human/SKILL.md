---
name: align-human
description: >
  Use when the user has a judge/grader and human-labeled data, and wants to measure
  how well the judge agrees with humans, detect systematic biases, determine whether
  automatic evaluation can replace human review, or build a human-reduction roadmap.
  Also use when the user mentions calibration, TPR/TNR, judge validation, inter-rater
  agreement, Cohen's kappa, bias detection, or "is my automatic evaluation trustworthy."
  Merges the calibrate and align functions into one skill.
---

<HARD-GATE>
NO calibrated:true WITHOUT TPR >= 0.8 AND TNR >= 0.8 AND boundary stratum TPR >= 0.6 AND TNR >= 0.6 AND n_dev >= 10 per class AND test-set drop < 10%.
NO human_reduction_phase >= 2 WITHOUT kappa >= 0.6 AND boundary kappa >= 0.6.
NO alignment conclusion WITHOUT all 5 bias checks completed.
</HARD-GATE>

# Align Human

Measure whether your automatic judge agrees with human judgment, detect where and
why they disagree, and build a roadmap to reduce human review over time.

## When to Activate

- You have a working judge/grader and 50+ human-labeled examples
- You want to know if the judge is trustworthy enough to replace human review
- You've noticed the judge's decisions being overturned by humans
- You're preparing to deploy an evaluation as a production gate

## Checklist

You MUST create a task for each item and complete them in order:

1. **Load paired data** — match judge verdicts with human labels
2. **Measure TPR/TNR** — confusion matrix + per-stratum breakdown
3. **Calculate agreement** — Cohen's kappa, Gwet's AC1, systematic bias
4. **Run bias detection** — 5 systematic bias checks
5. **Analyze disagreements** — cluster patterns + diagnose root causes
6. **Build human-reduction roadmap** — 4-phase transition plan
7. **Confirm and record** — one confirmation, then write results

## Fast path: run the bundled script

Don't hand-write the calibration statistics — that is exactly where subtle bugs hide. Run
the bundled, tested script (`scripts/calibration.py`, standard library only, **no OpenJudge
dependency**):

```bash
python scripts/calibration.py --pairs pairs.jsonl                 # one paired file, OR
python scripts/calibration.py --verdicts verdicts.jsonl --labels labels.jsonl --stratum-key difficulty
```

Paired rows look like `{"id","judge":"pass|fail","human":"pass|fail","stratum"?}` (judge/human
may also be 1/0). It prints the confusion matrix, TPR/TNR/F1 with bootstrap 95% CIs, Cohen's
kappa, Gwet's AC1 (auto-flags the kappa paradox), directional bias, per-stratum TPR/TNR, and
the **calibration gate** verdict (`calibrated` / `not_calibrated` / `insufficient_evidence`;
exit code 0 only if calibrated). `--json` for machine output, `--self-test` to verify it.

Always **report and interpret the actual numbers** the script returns — TPR/TNR (with their
95% CIs), Cohen's kappa, Gwet's AC1, directional bias, per-stratum TPR/TNR, and the gate
verdict — never just state that you ran it. If you don't yet have the paired verdicts/labels,
say exactly what's missing (e.g. the judge verdicts file, or N more labels per class).

The Steps below explain what each number means and how to act on it — read them to interpret
the script's output. The inline snippets are the reference behind the script; you normally
just run the script rather than re-implementing it.

## Step 1: Load and Pair Data

Human labels live in `labels/<grader_name>.jsonl`, one row per judged sample. Keep them
separate from the dataset so they can be re-paired with any judge run:

```python
{"id": "sample_017", "label": "pass",   # "pass"|"fail" (or 1/0); joins to a dataset row id
 "annotator": "alice", "rationale": "Order number matches context.",
 "timestamp": "2026-06-20T10:00:00Z", "schema_version": 1}
```

Match judge verdicts with human labels:

```python
import json

# Load human labels and judge verdicts
labels = {item["id"]: item["label"] for item in json.load(open("labels.jsonl"))}
verdicts = json.load(open("runs/verdicts-dev.jsonl"))

# Pair them
paired = []
for v in verdicts:
    if v["id"] in labels:
        paired.append({
            "id": v["id"],
            "judge": v["verdict"],   # "pass" or "fail"
            "human": labels[v["id"]], # "pass" or "fail"
        })

print(f"Paired: {len(paired)}, Unmatched: {len(verdicts) - len(paired)}")

# Warn if severe class imbalance
pass_rate = sum(1 for p in paired if p["human"] == "pass") / len(paired)
if pass_rate > 0.8 or pass_rate < 0.2:
    print(f"WARNING: Human label pass rate is {pass_rate:.0%} — "
          "kappa may be paradoxically low. Use Gwet's AC1 as complement.")
```

## Step 2: Measure TPR/TNR

Build a confusion matrix and compute per-stratum metrics:

```python
from openjudge.analyzer.validation import (
    AccuracyAnalyzer, F1ScoreAnalyzer,
    FalsePositiveAnalyzer, FalseNegativeAnalyzer,
)

# Convert to OpenJudge-compatible dataset with labels
analysis_dataset = [
    {"query": p.get("query", ""), "response": p.get("response", ""),
     "label": 1 if p["human"] == "pass" else 0}
    for p in paired
]

# Binary grader results (1=pass, 0=fail)
grader_results = [
    GraderScore(name="judge", score=1.0 if p["judge"] == "pass" else 0.0, reason="")
    for p in paired
]

accuracy = AccuracyAnalyzer().analyze(analysis_dataset, grader_results, label_path="label")
f1 = F1ScoreAnalyzer().analyze(analysis_dataset, grader_results, label_path="label")
fpr = FalsePositiveAnalyzer().analyze(analysis_dataset, grader_results, label_path="label")
fnr = FalseNegativeAnalyzer().analyze(analysis_dataset, grader_results, label_path="label")

# TPR = 1 - FNR, TNR = 1 - FPR
tpr = 1 - fnr.false_negative_rate
tnr = 1 - fpr.false_positive_rate
print(f"TPR={tpr:.2f}, TNR={tnr:.2f}, F1={f1.f1_score:.2f}")

# Per-stratum breakdown if difficulty data exists
for stratum in ["easy", "boundary", "hard"]:
    stratum_data = [d for d in analysis_dataset
                    if d.get("metadata", {}).get("difficulty") == stratum]
    if len(stratum_data) >= 10:
        # Compute TPR/TNR per stratum
        ...
```

Why per-stratum matters: A judge with TPR=0.9 overall but TPR=0.5 on boundary cases
is unreliable exactly where judgment matters most. This is the "progress illusion"
(EMNLP 2025) — aggregate metrics hide stratum-level failure.

### Bootstrap 95% CI

Use bootstrap resampling to quantify uncertainty:

```python
import numpy as np

def bootstrap_ci(samples, metric_fn, n_iter=1000, ci=95):
    """Compute bootstrap confidence interval for a metric."""
    n = len(samples)
    values = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        resampled = [samples[i] for i in idx]
        values.append(metric_fn(resampled))
    lower = np.percentile(values, (100 - ci) / 2)
    upper = np.percentile(values, 100 - (100 - ci) / 2)
    return np.mean(values), lower, upper

tpr_mean, tpr_low, tpr_high = bootstrap_ci(
    paired, lambda s: sum(1 for p in s if p["judge"] == "fail" and p["human"] == "fail")
                      / max(1, sum(1 for p in s if p["human"] == "fail"))
)
```

## Step 3: Calculate Agreement

### Cohen's Kappa (chance-corrected agreement)

```
kappa = (p_o - p_e) / (1 - p_e)
```
- p_o: observed agreement rate
- p_e: expected agreement by chance
- >= 0.8: substantial — judge is consistent with humans
- 0.6-0.8: moderate — conditional trust, needs spot-checking
- < 0.6: weak — cannot replace human judgment yet

### Gwet's AC1 (robust to class imbalance)

When 90% of samples are "pass," kappa can be paradoxically low even with high agreement.
Gwet's AC1 corrects for this. If kappa and AC1 differ by > 0.15, report both and note
the class imbalance effect.

### Systematic Bias

```
bias = P(judge=fail | human=pass) - P(judge=pass | human=fail)
```
- bias > 0.1: judge is stricter than humans (over-flagging)
- bias < -0.1: judge is more lenient than humans (under-flagging)
- |bias| < 0.1: no significant directional bias

## Step 4: Five Bias Detection Checks

| Check | What to look for | How to measure |
|-------|-----------------|----------------|
| **Position bias** | Does response order affect pairwise judgment? | Swap A/B order, compare win rates. Diff > 0.05 = bias |
| **Verbosity bias** | Do longer responses score higher? | Pearson r between score and response length. |r| > 0.3 = bias |
| **Self-enhancement** | Does the judge favor its own model family? | Check if judge model family = target model family. Same family = risk |
| **Progress illusion** | Does aggregate TPR hide boundary failure? | Compare overall TPR vs boundary TPR. Gap > 0.2 = illusion |
| **Label drift** | Have system outputs changed since labeling? | Compare historical vs current pass rate. Shift > 0.15 = drift |

## Step 5: Disagreement Pattern Analysis

For samples where judge and human disagree, cluster them to find root causes:

1. **Extract disagreement samples** — where judge verdict != human label
2. **Classify each disagreement**:
   - `judge_prompt_ambiguous`: pass/fail definitions not clear for this case
   - `human_inconsistent`: multiple human annotators disagreed on this sample
   - `task_inherently_subjective`: the dimension is fundamentally subjective
   - `label_error`: human label appears wrong, judge's reasoning is more convincing
   - `judge_too_strict`: judge applies criteria more harshly than humans intend
   - `judge_too_lenient`: judge overlooks issues humans catch

Present the top 3 patterns with 3 exemplars each so the user can decide whether
to refine the judge prompt or accept the disagreement as inherent noise.

## Step 6: Human-Reduction Roadmap

| Phase | Condition | Human role | Judge role | Trigger to advance |
|-------|-----------|-----------|------------|-------------------|
| **1: Advisory** | kappa < 0.6 | 100% human review | Judge is reference only | kappa >= 0.6 |
| **2: Assisted** | kappa >= 0.6 | Spot-check 20% of judgments | Judge is primary screener | kappa >= 0.8, boundary >= 0.6 |
| **3: Auto-gate** | kappa >= 0.8 | Review only borderline + low-confidence | Judge is production gate | kappa >= 0.9, all strata >= 0.8 |
| **4: Autonomous** | kappa >= 0.9 | Quarterly audit | Judge runs independently | Continuous monitoring |

## Step 7: Confirmation and Output

Present the alignment dashboard:

```
Alignment Results for [grader_name]:

TPR: 0.91  TNR: 0.88  F1: 0.90
Kappa: 0.87  AC1: 0.89  Bias: +0.03 (none)
95% CI (accuracy): [0.84, 0.93]

Per-stratum:
  easy:      TPR=0.96  TNR=0.94
  boundary:  TPR=0.82  TNR=0.79  ← weakest, monitor
  hard:      TPR=0.85  TNR=0.83

Bias checks:
  ✓ position:   no bias detected
  ✓ verbosity:  r=0.12 (clean)
  ✓ self-enh:   judge model != target model
  ✓ progress:   boundary gap 0.09 (acceptable)
  ✓ label drift: KL=0.08 (stable)

Phase: 3 (Auto-gate) — judge is calibrated and aligned

Disagreement patterns:
  - 12 samples: judge slightly stricter on multi-step queries
    Root cause: judge_prompt_ambiguous
    Recommendation: add a multi-step borderline example to few-shot

Recommendation: [calibrated + aligned | needs refinement | not ready]
```

## Common Mistakes

- **Using raw accuracy instead of TPR/TNR.** A judge that calls everything "pass"
  gets 90% accuracy when 90% of samples pass, but catches zero failures. TPR and
  TNR decompose accuracy into what actually matters.
- **Trusting kappa without checking class balance.** With 95% pass rate, kappa
  can be 0.4 even with 95% raw agreement. Always report Gwet's AC1 alongside kappa.
- **Skipping per-stratum analysis.** Aggregate TPR of 0.9 with boundary TPR of 0.5
  means the judge is unreliable exactly where you need it most.
- **Not setting a stop condition for iteration.** Refining the judge prompt has
  diminishing returns. After 3 iterations with no kappa improvement > 0.03, stop
  and collect more labeled data instead.
- **Forgetting judge model != target model constraint.** Self-evaluation inflates
  TPR by 3-8%. Always verify these are different models.

## Next Skills

After `03-align-human`:
- **`04-eval-report`**: Generate a comprehensive report with maturity assessment.
- **`02-metric-design`**: If you need to redesign graders based on bias findings.
- **`01-eval-design`**: If disagreement patterns reveal dataset coverage gaps.