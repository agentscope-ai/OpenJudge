---
name: prompt-regression
description: >
  Use when the user has changed a prompt (system prompt, RAG template, agent instruction,
  etc.) and wants to know whether the candidate is better or worse than the baseline.
  Also use when the user mentions prompt A/B testing, prompt comparison, prompt
  optimization validation, "did my prompt change help," or prompt regression testing.
  Outputs per-dimension win rates with statistical significance using OpenJudge
  PairwiseAnalyzer.
---

<HARD-GATE>
NO conclusion about which prompt is better WITHOUT bootstrap 95% CI reported.
NO candidate declared "better" WITHOUT position-debiased (swap-aggregate) comparison.
NO comparison with fewer than 10 samples per axis — CI is too wide to be meaningful.
</HARD-GATE>

# Prompt Regression

Compare two prompts head-to-head and determine, with statistical rigor, whether
the candidate is better, worse, or tied on each evaluation dimension.

## When to Activate

- You changed the system prompt and want to verify it's actually better
- You're iterating on RAG answer templates
- You're optimizing agent step-by-step instructions
- You want data to support a prompt change decision

## Checklist

You MUST create a task for each item and complete them in order:

1. **Load and analyze prompts** — diff the baseline vs candidate
2. **Derive comparison dimensions** — from the prompt changes + task type
3. **Select graders per dimension** — pairwise, judge, or rule
4. **Run position-debiased comparison** — swap-aggregate to eliminate order bias
5. **Compute statistics** — win rates + bootstrap 95% CI per dimension
6. **Present results** — per-dimension verdict with confidence intervals

## Fast path: run the bundled script

Don't hand-write the win-rate + bootstrap math (the swap-aggregation and CI are easy to get
wrong). Run the bundled, tested script (`scripts/pairwise.py`, standard library only, **no
OpenJudge dependency**):

```bash
python scripts/pairwise.py --comparisons comparisons.jsonl --candidate candidate --baseline baseline
```

Each comparison row: `{"id","model_a","model_b","score","dimension"?}` where `score >= 0.5`
means `model_a` won. Emit two rows per query with A/B **swapped** to debias position. The
script reports per-dimension candidate/baseline/tie rates, bootstrap 95% CI, and a verdict
(`BETTER` / `WORSE` / `TIED` / `INSUFFICIENT_EVIDENCE` / `INCONCLUSIVE`; exit 0 only if
better). `--self-test` to verify it.

Steps below explain how to derive dimensions and produce the comparisons (with OpenJudge or
any judge); the inline snippets are the reference behind the script.

## Step 1: Load and Analyze Prompts

Read the baseline and candidate prompts. Identify:
- **Task type**: chatbot / RAG generation / code review / translation / summarization
  / agent instruction / other
- **What changed**: added constraints, changed tone, new examples, different output
  format, expanded/shortened instructions
- **Intent of change**: what problem was the user trying to fix?

## Step 2: Derive Comparison Dimensions

Based on the task type and what changed, derive 3-5 comparison dimensions.

### Dimension templates by task type

**Chatbot / Conversational**:
- Answer relevance — does it address the user's question?
- Tone appropriateness — does the tone match context?
- Factual accuracy — no fabricated information
- Conciseness — doesn't ramble or over-explain
- Instruction following — obeys system prompt constraints

**RAG Generation**:
- Faithfulness — grounded in retrieved documents
- Citation accuracy — correctly references sources
- Completeness — covers all aspects of the query
- No hallucination — no claims beyond documents

**Code Review / Generation**:
- Bug detection — finds real issues
- False positive rate — doesn't flag correct code
- Actionability — suggestions are specific and implementable
- Code style — follows conventions

**Agent Instructions**:
- Tool selection — picks the right tool
- Step efficiency — minimal steps to goal
- Error recovery — handles failures gracefully
- Output format — follows specified structure

Each dimension gets:
- An `id` (slug)
- A one-sentence description
- A grader type: `pairwise` or `judge` or `rule`

## Step 3: Select Graders

Decision priority:
1. **Can a rule check this?** → `FunctionGrader` or `StringMatchGrader`. Free, deterministic.
   Example: output length, keyword presence, JSON validity.
2. **Is there a reference answer?** → `pairwise` against reference.
3. **Subjective quality, no reference?** → `pairwise` A/B comparison.
4. **Single-output judgment needed?** → `judge` (binary pass/fail per output).

## Step 4: Run Position-Debiased Comparison

### Pairwise comparison with swap-aggregate

LLM judges have position bias — the first response shown wins 5-15% more often.
Swap-aggregate eliminates this: run each comparison twice with swapped positions,
keep only consistent wins:

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.runner.grading_runner import GradingRunner
from openjudge.analyzer.pairwise_analyzer import PairwiseAnalyzer

# Judge prompt for relevance comparison
relevance_judge = LLMGrader(
    model=model,
    name="relevance_compare",
    mode=GraderMode.POINTWISE,
    template="""
Compare Response A and Response B for the query below.
Which response better addresses the user's question?

Query: {query}
Response A: {response_a}
Response B: {response_b}

Score 1.0 if A is better, 0.0 if B is better, 0.5 if tied.
Respond in JSON: {{"score": <float>, "reason": "<explanation>"}}
""",
)

# Build pairwise dataset with position swap
dataset = []
for sample in test_samples:
    # Original order
    dataset.append({
        "query": sample["query"],
        "response_a": baseline_outputs[sample["id"]],
        "response_b": candidate_outputs[sample["id"]],
        "metadata": {"model_a": "baseline", "model_b": "candidate"},
    })
    # Swapped order — critical for debiasing
    dataset.append({
        "query": sample["query"],
        "response_a": candidate_outputs[sample["id"]],
        "response_b": baseline_outputs[sample["id"]],
        "metadata": {"model_a": "candidate", "model_b": "baseline"},
    })

runner = GradingRunner(
    grader_configs={"relevance": relevance_judge},
    max_concurrency=8,
)
results = await runner.arun(dataset)

# Analyze with PairwiseAnalyzer
analyzer = PairwiseAnalyzer(model_names=["baseline", "candidate"])
analysis = analyzer.analyze(dataset, results["relevance"])

print(f"Win rates: {analysis.win_rates}")
# → {'baseline': 0.35, 'candidate': 0.55} → candidate wins 55% of comparisons
print(f"Best model: {analysis.best_model}")
```

Why swap-aggregate? Without it, if the judge prefers the first response shown,
and you always show baseline first, you'll systematically underrate the candidate.

## Step 5: Compute Statistics

For each dimension, report:
- Candidate win rate, baseline win rate, tie rate
- Bootstrap 95% confidence interval
- Verdict: better / worse / tied / inconclusive

`PairwiseAnalyzer.analyze` interprets each comparison as `score >= 0.5 → model_a wins`,
using the row's `metadata.model_a` / `metadata.model_b`. So derive a per-comparison
winner list from `dataset` + `results`, then bootstrap over that list — never index the
`PairwiseAnalysisResult` object (it has no per-sample rows).

```python
import numpy as np
from openjudge.graders.schema import GraderScore

def per_comparison_winners(dataset, grader_results):
    """One named winner per comparison row (handles swapped order via metadata)."""
    winners = []
    for sample, result in zip(dataset, grader_results):
        if not isinstance(result, GraderScore):
            continue  # skip errors
        meta = sample.get("metadata", {})
        winners.append(meta["model_a"] if result.score >= 0.5 else meta["model_b"])
    return winners

def bootstrap_win_rate(winners, target, n_iter=1000):
    n = len(winners)
    rates = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        rates.append(sum(1 for i in idx if winners[i] == target) / n)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))

winners = per_comparison_winners(dataset, results["relevance"])
n = len(winners)
candidate_rate = sum(1 for w in winners if w == "candidate") / n
baseline_rate = sum(1 for w in winners if w == "baseline") / n
ci_low, ci_high = bootstrap_win_rate(winners, target="candidate")

if ci_low > 0.5:
    verdict = "candidate BETTER"
elif ci_high < 0.5:
    verdict = "candidate WORSE"
elif (ci_high - ci_low) < 0.3:
    verdict = "TIED (CI brackets 0.5, narrow)"
else:
    verdict = "INCONCLUSIVE (CI too wide — need more samples)"

print({"candidate_win_rate": candidate_rate, "baseline_win_rate": baseline_rate,
       "ci_95": [ci_low, ci_high], "verdict": verdict})
```

Note: with swap-aggregate each query produces 2 comparison rows. Bootstrapping over rows
(above) is the simple approach; for a tighter estimate, bootstrap over *queries* and
average the 2 swapped rows per query so position pairs stay together.

## Step 6: Present Results

```
Prompt Regression: v1 (baseline) vs v2 (candidate)
Task: Customer support chatbot
Samples: 50

Dimension            Candidate  Baseline  Tie   95% CI         Verdict
===========================================================================
Answer relevance        58%       32%      10%   [51%, 65%]   ✓ BETTER
Factual accuracy        48%       44%       8%   [41%, 55%]   = TIED
Tone appropriateness    38%       52%      10%   [31%, 45%]   ✗ WORSE
Conciseness             62%       28%      10%   [55%, 69%]   ✓ BETTER

Summary: v2 is significantly better on relevance and conciseness,
but worse on tone appropriateness. The tone regression likely comes
from the new "be direct" instruction — consider softening it.

Top 3 tone failures (candidate worse):
  1. Query: "I'm really frustrated..." → v2 response too curt
  2. Query: "This is my first time..." → v2 missing empathetic opening
  3. Query: "Can you help me understand..." → v2 skipped explanation
```

## Common Mistakes

- **Not doing position swap.** Position bias in LLM judges is 5-15%. Without
  swap-aggregate, results are systematically skewed.
- **Comparing with < 10 samples.** Bootstrap CI at n=10 is ±15%+ half-width.
  At n=5 it's ±25%+. Results are noise, not signal. Minimum 10, prefer 30+.
- **Single "overall" comparison without dimensions.** "V2 is 55% better" hides
  that it's +20% on relevance but -15% on tone. Always report per-dimension.
- **Accepting ties as "no difference."** A true tie and insufficient data look
  identical without CI. Always report confidence intervals.
- **Not pinning model versions.** If baseline and candidate are run on different
  model versions (even same model, different date), model drift contaminates
  the prompt comparison. Same model, same version, same temperature.

## Next Skills

After `06-prompt-regression`:
- **`03-align-human`**: Calibrate the pairwise judge against human preferences.
- **`02-metric-design`**: Turn validated dimensions into permanent graders.
- **`04-eval-report`**: Include prompt comparison results in a comprehensive report.