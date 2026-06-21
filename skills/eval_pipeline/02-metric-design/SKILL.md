---
name: metric-design
description: >
  Use when the user has evaluation principles or a dataset but needs help choosing
  the right graders, designing evaluation metrics, creating LLM-as-judge prompts,
  combining multiple metrics into a composite score, or building an automated
  evaluation pipeline. Also use when the user mentions grader selection, metric
  design, judge prompt engineering, rubric design, evaluation pipeline code,
  or "how to evaluate [X] automatically." Outputs executable OpenJudge pipeline code.
---

# Metric Design

Select, configure, and combine evaluation graders into a working pipeline. You choose
the right tool for each evaluation dimension — from zero-cost code checks to LLM judges
— and produce executable `GradingRunner` code that runs on OpenJudge.

> **Requires OpenJudge** (`pip install py-openjudge`). This skill is intentionally
> SDK-centric — grader selection, `GradingRunner`, and aggregators are OpenJudge APIs. The
> design/decision logic still applies if you use another harness; only the code does not.

## When to Activate

- User has eval dimensions/principles but doesn't know which grader type to use
- User wants to write an LLM-as-judge prompt for a specific failure mode
- User needs a composite score combining multiple evaluation dimensions
- User wants to auto-generate graders from labeled data instead of writing them manually
- User's current evaluation is all LLM-based and too expensive/too slow

## Checklist

You MUST create a task for each item and complete them in order:

1. **Select grader types** — per dimension, pick the right grader class
2. **Create custom graders** — write judge prompts (4-component) or function graders
3. **Auto-generate if applicable** — use OpenJudge Generator for cold starts
4. **Run anti-pattern scan** — check for Likert, missing few-shot, vague criteria
5. **Build pipeline code** — assemble GradingRunner with graders + aggregators

## Step 1: Select Grader Type Per Dimension

For each evaluation dimension, walk this decision tree (first match wins):

```
1. Can a deterministic rule check this?
   → StringMatchGrader / JsonValidatorGrader / FunctionGrader (zero cost, 100% consistent)
   Examples: exact match for classification labels, regex for format checks,
             JSON schema validation, keyword presence/absence

2. Does it require semantic understanding of text quality?
   → LLMGrader with built-in class (low cost, pre-optimized)
   Examples: CorrectnessGrader (factual match), RelevanceGrader (on-topic check),
             HallucinationGrader (faithfulness to context)

3. Does it involve agent behavior (tool calls, planning, memory)?
   → Agent-specific LLMGrader
   Examples: ToolSelectionGrader, TrajectoryAccuracyGrader, MemoryAccuracyGrader

4. Does it involve code execution or syntax?
   → CodeExecutionGrader / SyntaxCheckGrader
   Examples: test case pass rate, syntax validity, code style checks

5. Does it require external tool calls to verify (web search, database lookup)?
   → AgenticGrader (expensive, use only when necessary)
   Examples: fact-checking against live sources, cross-referencing databases
```

### Grader Selection Cheat Sheet

| Output type | Recommended grader | Cost |
|------------|-------------------|------|
| Classification label | `StringMatchGrader` | Free |
| JSON structure | `JsonValidatorGrader` + `JsonMatchGrader` | Free |
| Free text correctness | `CorrectnessGrader` | LLM call |
| Factual accuracy (grounded) | `HallucinationGrader` | LLM call |
| Response relevance | `RelevanceGrader` | LLM call |
| Instruction following | `InstructionFollowingGrader` | LLM call |
| Tool call selection | `ToolSelectionGrader` | LLM call |
| Agent trajectory | `TrajectoryAccuracyGrader` | LLM call |
| Code correctness | `CodeExecutionGrader` | Free |
| Custom quality check | Custom `LLMGrader` | LLM call |
| External fact verification | `AgenticGrader` | LLM + tool calls |

**Why this order matters**: Every LLM-based grader adds cost, latency, and non-determinism.
A `StringMatchGrader` costs nothing and always gives the same answer. Exhaust deterministic
options before reaching for an LLM judge.

## Step 2: Create Custom Graders

### LLMGrader: The Four-Component Template

When no built-in grader fits, create a custom `LLMGrader`. Every judge prompt needs
exactly these four components (adapted from community best practice):

**Component 1 — Task & Criterion**: What this judge evaluates. One thing only.

```
You are evaluating whether a customer support response correctly identifies
and uses the customer's order number from the conversation context.
```

**Component 2 — Binary Pass/Fail Definitions**: Concrete, observable conditions.

```
PASS: The response references the correct order number exactly as it appears
in the context. If multiple orders exist, the response addresses the right one.

FAIL: The response uses a wrong order number, omits the order number when one
was provided, or references an order not present in the context.
```

Why binary and not Likert? Because two human annotators agree on "pass vs fail" far more
often than on "3 vs 4 out of 5." Binary forces a clear decision boundary. If you need
severity levels, use multiple binary judges (e.g., "factually wrong" + "dangerously wrong").

**Component 3 — Few-Shot Examples**: At minimum 1 pass, 1 fail, 1 borderline.
The borderline example is the most valuable — it teaches the judge where the boundary is.

```
Example 1 (PASS):
Context: "Order #12345: shipped May 10"
Response: "Your order #12345 was shipped on May 10 and arrives May 12."
Critique: The response uses the exact order number (#12345) and matches the
ship date from context. No fabrication or omission.
Result: Pass

Example 2 (FAIL):
Context: "Order #12345: shipped May 10"
Response: "Your order #12346 is on its way!"
Critique: The response uses order #12346 but the context only mentions #12345.
This is a fabricated order number, not a typo — #12346 doesn't exist.
Result: Fail

Example 3 (BORDERLINE PASS):
Context: "Orders #12345 (shipped), #12346 (processing)"
Response: "Your recent order has shipped and should arrive soon."
Critique: The response doesn't specify which order, but says "recent order"
which could reasonably refer to either. If the customer only asked about
shipped items, this is fine. If they asked about a specific order, it's
insufficient. Given the generic phrasing, this passes but is weak.
Result: Pass
```

**Component 4 — Structured Output**: Force `critique` before `verdict`.

```json
{
  "critique": "Detailed assessment referencing specific evidence from the response and context",
  "result": "Pass" or "Fail"
}
```

Why critique-before-verdict? LLMs that commit to a verdict first anchor on it and
rationalize backward. Reasoning first → verdict second produces more accurate judgments
(CoT-then-Score AUC ~0.97 vs verdict-first significantly lower).

### Complete LLMGrader Code

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode

order_accuracy_grader = LLMGrader(
    model=model,
    name="order_accuracy",
    mode=GraderMode.POINTWISE,
    template="""
You are evaluating whether a customer support response correctly identifies
and uses the customer's order number from the conversation context.

Context: {context}
Response: {response}

## Pass/Fail Definitions

PASS: The response references the correct order number exactly as it appears
in the context. If multiple orders exist, the response addresses the right one.

FAIL: The response uses a wrong order number, omits the order number when one
was provided, or references an order not present in the context.

## Examples

Example 1 (PASS):
Context: "Order #12345: shipped May 10"
Response: "Your order #12345 was shipped on May 10 and arrives May 12."
Critique: Exact order number match. Ship date matches context. No fabrication.
Result: Pass

Example 2 (FAIL):
Context: "Order #12345: shipped May 10"
Response: "Your order #12346 is on its way!"
Critique: Order #12346 does not exist in context. Fabricated order number.
Result: Fail

Example 3 (BORDERLINE PASS):
Context: "Orders #12345 (shipped), #12346 (processing)"
Response: "Your recent order has shipped and should arrive soon."
Critique: Doesn't specify which order. "Recent order" is ambiguous but not
factually wrong — it acknowledges a shipped order exists.
Result: Pass

## Output Format

Respond in JSON:
{{"critique": "<detailed assessment>", "result": "Pass" or "Fail"}}
""",
)
```

### FunctionGrader: Deterministic Checks

Use when the rule is code-expressible:

```python
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore, GraderMode

def no_competitor_mention(response: str, competitors: list[str] = None) -> GraderScore:
    """Check that response doesn't mention competitor brands."""
    if competitors is None:
        competitors = ["competitor_a", "competitor_b", "rival_co"]
    mentioned = [c for c in competitors if c.lower() in response.lower()]
    if not mentioned:
        return GraderScore(name="no_competitor", score=1.0, reason="No competitor mentions")
    return GraderScore(
        name="no_competitor", score=0.0,
        reason=f"Mentioned competitors: {', '.join(mentioned)}"
    )

competitor_grader = FunctionGrader(
    func=no_competitor_mention,
    name="no_competitor",
    mode=GraderMode.POINTWISE,
)
```

## Step 3: Auto-Generate Graders (Cold Start)

When you have no rubric but do have a task description or labeled data, use OpenJudge
Generators to create graders automatically:

### Zero-shot: SimpleRubricsGenerator

```python
from openjudge.generator.simple_rubric.generator import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
)

config = SimpleRubricsGeneratorConfig(
    grader_name="Customer Support Quality",
    model=model,
    task_description="Customer support chatbot for e-commerce: orders, returns, shipping",
    scenario="Customers asking about order status, return policies, and delivery times",
    min_score=0,
    max_score=1,
)

generator = SimpleRubricsGenerator(config)
grader = await generator.generate(
    dataset=[],
    sample_queries=[
        "Where is my order?",
        "How do I return this item?",
        "When will my package arrive?",
    ],
)
# grader is now a ready-to-use LLMGrader
```

### Data-driven: IterativeRubricsGenerator

Use when you have 20+ labeled examples (query + response + score):

```python
from openjudge.generator.iterative_rubric.generator import (
    IterativeRubricsGenerator,
    IterativePointwiseRubricsGeneratorConfig,
)

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="E-commerce QA Grader",
    model=model,
    task_description="Evaluate factual answers to e-commerce customer questions",
    min_score=0,
    max_score=1,
    max_epochs=3,
    batch_size=10,
)

train_data = [
    {"query": "What's your return policy?", "response": "30-day returns, free shipping.", "label_score": 1},
    {"query": "What's your return policy?", "response": "We have a policy.", "label_score": 0},
    # ... 20+ examples
]

generator = IterativeRubricsGenerator(config)
grader = await generator.generate(dataset=train_data)
```

## Step 4: Anti-Pattern Scan

Before finalizing, check every LLM-based grader for these issues:

| Check | What to look for | Severity |
|-------|-----------------|----------|
| Likert scale | "rate 1-5", "score 1-10", "Likert" in prompt | BLOCKER — replace with binary Pass/Fail |
| Missing few-shot | No labeled examples in the prompt | BLOCKER — add at least 1 pass + 1 fail + 1 borderline |
| Holistic criterion | Single judge evaluating 3+ dimensions | WARNING — split into separate graders, one per dimension |
| Missing output format | No JSON schema specified | BLOCKER — add {{"critique": "...", "result": "Pass"/"Fail"}} |
| Vague pass/fail | < 20 words or uses "good"/"bad"/"quality" | WARNING — make definitions concrete and observable |
| Judge = target model | Same model for both roles | BLOCKER — judge and target must be different models |

**Why blockers matter**: A Likert-scale judge with no few-shot examples and a vague
criterion produces scores that look precise but can't be reproduced or calibrated.
You'll discover this in production when the judge's TPR/TNR is measured — and it's too late.

## Step 5: Build Pipeline Code

First record the design as a `metric-plan.yaml` so it's reusable and reviewable, then
implement it as the runner below:

```yaml
dimensions:
  - {id: order_accuracy, grader: CorrectnessGrader, mode: score, weight: 0.4,
     mapper: {response: response, reference_response: reference_response}}
  - {id: no_pii, grader: FunctionGrader, mode: gate, gate_threshold: 1.0}  # hard requirement
aggregation: GatedWeightedSumAggregator
```

Assemble everything into a working `GradingRunner`:

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.graders.text.string_match import StringMatchGrader
from openjudge.runner.grading_runner import GradingRunner, GraderConfig
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.graders.schema import GraderScore, GraderError

# Judge model (must differ from the model being evaluated).
# OpenAIChatModel reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment when
# not passed explicitly — point them at any OpenAI-compatible endpoint.
#   OpenAI:          OPENAI_API_KEY=sk-...   (no base_url needed)
#   Aliyun DashScope: OPENAI_API_KEY=<dashscope key>
#                     OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
model = OpenAIChatModel(model="qwen-plus")  # or "gpt-4o", etc.

dataset = [
    {
        "query": "Where is my order #12345?",
        "response": "Your order #12345 shipped May 10, arriving May 12.",
        "reference_response": "Order #12345: shipped May 10, ETA May 12. Tracking: 1Z999AA10123456784.",
        "context": "Order #12345 | Status: shipped | Date: 2026-05-10 | Tracking: 1Z999AA10123456784",
    },
    # ... more samples
]

runner = GradingRunner(
    grader_configs={
        # LLM-based graders for semantic quality
        "correctness": CorrectnessGrader(model=model),
        "relevance": RelevanceGrader(model=model),
        "hallucination": HallucinationGrader(model=model),
        # Deterministic grader — zero cost
        "format_json": GraderConfig(
            grader=StringMatchGrader(algorithm="substring_match"),
            mapper={"response": "response", "reference_response": "reference_response"},
        ),
    },
    # Combine into a single weighted score
    aggregators=WeightedSumAggregator(
        name="overall",
        weights={
            "correctness": 0.4,
            "relevance": 0.2,
            "hallucination": 0.3,
            "format_json": 0.1,
        },
    ),
    max_concurrency=8,
)

async def main():
    results = await runner.arun(dataset)

    for grader_name, grader_results in results.items():
        scores = [r.score for r in grader_results if isinstance(r, GraderScore)]
        errors = [r for r in grader_results if isinstance(r, GraderError)]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{grader_name}: avg={avg:.3f}, errors={len(errors)}")

asyncio.run(main())
```

### Weight Design Principle

Don't use equal weights — they're the most arbitrary choice. Weights should reflect:

- **Failure prevalence** (from trace analysis): if `hallucination` failures occur 3x
  more often than `relevance` failures, weight hallucination higher.
- **Business impact**: a correctness failure might cost a customer; a tone failure might
  slightly annoy them. Weight accordingly.
- **Gating vs scoring**: safety/correctness dimensions should be conjunctive gates
  (must pass), not weighted scores (can be compensated by other dimensions). See the
  gate pattern below — do not fold a hard requirement (PII, safety, legal) into a
  `WeightedSumAggregator`, because a high score elsewhere can mask the violation.

### Gates vs weighted scores (conjunctive requirements)

A `WeightedSumAggregator` lets dimensions compensate each other: a perfect JSON score
can drag a PII leak up to "passing." For any requirement that must **never** be traded
off (PII, safety, legal compliance), implement a **gate** — if it fails, the whole sample
fails regardless of the other scores. OpenJudge ships only `WeightedSumAggregator`, so
write a tiny gate aggregator:

```python
from typing import Dict
from openjudge.runner.aggregator.base_aggregator import BaseAggregator
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.graders.schema import GraderResult, GraderScore

class GatedWeightedSumAggregator(BaseAggregator):
    """Weighted sum that hard-fails (score=0.0) if any gate grader is below threshold.

    gate_graders: grader names that act as conjunctive gates (must pass).
    A gate 'passes' when its score >= gate_threshold.
    """

    def __init__(self, name: str, weights: Dict[str, float],
                 gate_graders: list[str], gate_threshold: float = 1.0):
        super().__init__(name)
        self.gate_graders = gate_graders
        self.gate_threshold = gate_threshold
        # Score only the non-gate dimensions; gates are pass/fail, not weighted.
        self._scorer = WeightedSumAggregator(
            name=name,
            weights={k: v for k, v in weights.items() if k not in gate_graders},
        )

    def __call__(self, grader_results: Dict[str, GraderResult], **kwargs) -> GraderResult:
        for gate in self.gate_graders:
            res = grader_results.get(gate)
            if isinstance(res, GraderScore) and res.score < self.gate_threshold:
                return GraderScore(
                    name=self.name, score=0.0,
                    reason=f"GATE FAILED: {gate}={res.score} (< {self.gate_threshold}). "
                           f"Hard requirement violated; weighted score suppressed.",
                    metadata={"gate_failed": gate},
                )
        # All gates passed → weighted sum of the remaining (compensable) dimensions.
        return self._scorer(grader_results, **kwargs)

# Example: PII leakage is a gate; JSON-field correctness is the compensable score.
runner = GradingRunner(
    grader_configs={
        "fields_present": fields_grader,   # claim_id/status/amount present (deterministic)
        "no_pii": no_pii_grader,           # GATE: 1.0 = clean, 0.0 = PII present
    },
    aggregators=GatedWeightedSumAggregator(
        name="overall",
        weights={"fields_present": 1.0},   # only non-gate dims are weighted
        gate_graders=["no_pii"],           # PII leak ⇒ overall 0.0 no matter what
    ),
    max_concurrency=8,
)
```

Rule of thumb: if a stakeholder would say "I don't care how good the rest is, this can
never ship if X happens," then X is a gate, not a weight.

## Common Mistakes

- **All LLM judges, no deterministic checks.** Every LLM call adds cost and noise.
  ~30-50% of evaluation dimensions can be checked with code. Check those first.
- **One judge evaluating 3+ things.** A single holistic judge produces unactionable
  verdicts. "The response scored 3/5" tells you nothing about what to fix. Split into
  one judge per dimension.
- **Likert scales.** "Rate helpfulness 1-5" produces scores that look scientific but
  can't be calibrated — annotators disagree on 3 vs 4 far more than pass vs fail.
- **No few-shot examples.** Without examples, the judge model guesses what "pass" means
  in your context. The borderline example is the most important one.
- **Judge uses the same model as the target.** Self-evaluation bias inflates scores.
  Always use a different model (or at minimum a different model version).
- **Equal weights for composite scores.** Equal weights = "I don't know what matters."
  Derive weights from failure prevalence or business impact.

## Next Skills

After `02-metric-design`:
- **`03-align-human`**: You have graders. Now calibrate TPR/TNR against human labels
  to know if the automatic evaluation is trustworthy.
- **`04-eval-report`**: Run evaluation at scale and generate analysis with
  OpenJudge's DistributionAnalyzer.