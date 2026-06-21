---
name: bootstrap
description: >
  Use when the user has nothing — no traces, no labels, no eval set — and needs to
  build a v0 evaluation from scratch. Also use when the user says "I need to start
  evaluating my app but don't know where to begin," "I want to set up eval for a new
  product," or has just identified failure modes and needs to turn them into principles.
  Outputs a v0 grader in 30 minutes using OpenJudge SimpleRubricsGenerator, plus a
  roadmap to reach calibrated evaluation.
---

<HARD-GATE>
NO v0 grader deployed WITHOUT explicitly marking it as uncalibrated.
NO synthetic labels — LLM can generate eval inputs, but labels MUST come from real system output + human judgment.
NO principle without a source label documenting where it came from.
</HARD-GATE>

# Bootstrap

Cold-start an evaluation system when you have nothing. In 30 minutes you get a working
v0 grader and a clear path to a calibrated, trustworthy evaluation.

> **Requires OpenJudge** (`pip install py-openjudge`) for the grader generators
> (`SimpleRubricsGenerator` / `IterativeRubricsGenerator`). The interview, stratification,
> and calibration-roadmap methodology is SDK-independent.

## Checklist

You MUST create a task for each item and complete them in order:

1. **Understand the product** — one-shot interview, not question-by-question
2. **Generate v0 grader** — use OpenJudge SimpleRubricsGenerator
3. **Synthesize eval inputs** — 30 inputs with 60/30/10 stratification
4. **Run v0 evaluation** — GradingRunner with the generated grader
5. **Output roadmap** — exactly how to reach 50 labels → calibrate

## Step 1: Product Interview (One Shot)

Ask the user to describe their system in one go:

```
To bootstrap your evaluation, I need to understand what you're building.
Please describe (all at once):

- What does your system do? Who uses it?
- What are 3 examples of perfect outputs?
- What are 3 things the system must never do?
- What failures worry you most?
```

Don't drip-feed these questions. One prompt, one answer. If the user provides a spec
doc or design document instead, read that directly.

## Step 2: Generate v0 Grader

Use OpenJudge's `SimpleRubricsGenerator` to create a zero-shot grader from the
product description:

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.generator.simple_rubric.generator import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
)
from openjudge.runner.grading_runner import GradingRunner

# OpenAIChatModel reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment.
# For Aliyun DashScope (Bailian): set OPENAI_BASE_URL to
# https://dashscope.aliyuncs.com/compatible-mode/v1 and OPENAI_API_KEY to your key.
model = OpenAIChatModel(model="qwen-plus")  # or "gpt-4o", etc.

config = SimpleRubricsGeneratorConfig(
    grader_name="Initial Quality Grader",
    model=model,
    task_description="<summarize from the interview>",
    scenario="<usage context from interview>",
    min_score=0,
    max_score=1,
)

generator = SimpleRubricsGenerator(config)
grader = await generator.generate(
    dataset=[],
    sample_queries=[
        "<example query 1 from interview>",
        "<example query 2 from interview>",
        "<example query 3 from interview>",
    ],
)
```

Why zero-shot instead of asking the user to write criteria? At this stage, the user
doesn't know what "good" means operationally. The generator produces a reasonable
starting point. The user refines it after seeing v0 results.

## Step 3: Synthesize Eval Inputs

Generate 30 test inputs with stratification. Use 3 different prompt templates for
diversity:

```
Template 1: "Generate a typical {domain} query for a {user_type}"
Template 2: "Create an ambiguous {domain} query where intent is unclear"
Template 3: "Generate an edge-case {domain} query that's unusual but realistic"
```

Target distribution:
- 60% common/typical queries (18 inputs)
- 30% boundary/ambiguous queries (9 inputs)
- 10% edge-case/unusual queries (3 inputs)

**Critical**: Generate inputs ONLY. Never generate labels. The labels come from
running the actual system and getting human judgments.

```python
# The dataset format for GradingRunner
dataset = [
    {
        "query": "What's the status of my order #12345?",
        "response": "<will be filled by running the system>",
    },
    # ... 30 inputs
]
```

## Step 4: Run v0 Evaluation

Plug the generated grader into GradingRunner:

```python
from openjudge.runner.grading_runner import GradingRunner
from openjudge.graders.schema import GraderScore, GraderError

runner = GradingRunner(
    grader_configs={"v0_quality": grader},
    max_concurrency=8,
)

results = await runner.arun(dataset)

scores = [r.score for r in results["v0_quality"] if isinstance(r, GraderScore)]
errors = [r for r in results["v0_quality"] if isinstance(r, GraderError)]
print(f"V0 Results: avg={sum(scores)/len(scores):.2f}, errors={len(errors)}")
```

## Step 5: Output Roadmap

The v0 grader is uncalibrated — you don't know its TPR/TNR yet. Give the user an
exact path to trustworthiness:

```
Your v0 evaluation is ready. Here's the path to a calibrated system:

Phase 1 (now): Run the v0 grader on 30 inputs to get a baseline.
  → The grader is UNCALIBRATED. Treat scores as directional, not definitive.

Phase 2 (1-2 weeks): Collect 50 human-labeled examples (25 pass + 25 fail).
  → For each system output, have a human mark pass/fail against the criterion.
  → Store labels in labels/<grader_name>.jsonl

Phase 3: When you have 50 labels, run 03-align-human to:
  → Measure TPR/TNR of the v0 grader
  → Detect biases (position, verbosity, self-enhancement)
  → Get a human-reduction roadmap

Phase 4: When TPR >= 0.8 and TNR >= 0.8:
  → The grader is calibrated and can be used as a production gate
```

## Quick Mode vs Deep Mode

- **Quick mode (default)**: Steps 1-5 above. 30 minutes to v0. Use when stakes=low
  or when exploring.
- **Deep mode**: If the user has 20+ labeled examples, use `IterativeRubricsGenerator`
  instead of `SimpleRubricsGenerator` for data-driven grader creation:

```python
from openjudge.generator.iterative_rubric.generator import (
    IterativeRubricsGenerator,
    IterativePointwiseRubricsGeneratorConfig,
)

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="Data-Driven Grader",
    model=model,
    task_description="<from interview>",
    min_score=0, max_score=1,
    max_epochs=3,
    batch_size=10,
)
generator = IterativeRubricsGenerator(config)
grader = await generator.generate(dataset=labeled_data)  # 20+ labeled examples
```

## Red Flags — STOP and Re-evaluate

- "I'll generate both inputs and labels with the LLM to save time" → STOP.
  LLM-generating labels creates a self-consistency loop. TPR will look great
  until you test on real data, then it collapses.
- "The v0 grader looks good, let's deploy it as a gate" → STOP. Uncalibrated
  graders have unknown TPR/TNR. They might pass everything or fail everything.
- "I'll skip the roadmap, the user knows what to do next" → STOP. The roadmap
  IS the deliverable. Without it, bootstrap just produces an untrustworthy grader.

## Common Mistakes

- **Over-interviewing**. One prompt with 4 questions. Don't ask follow-ups unless
  the answers are genuinely unclear.
- **Too many principles in v0**. SimpleRubricsGenerator works best with a focused
  task description. Don't try to evaluate 10 dimensions in v0 — start with the
  2-3 most important ones.
- **Skipping stratification in synthetic inputs**. If all 30 inputs are typical
  queries, you'll never see how the system handles edge cases.
- **Presenting v0 scores as truth**. Always prefix v0 results with "UNVERIFIED —
  these scores are directional only."

## Next Skills

After `08-bootstrap`:
- **`03-align-human`**: Once 50 human labels are collected, calibrate the grader.
- **`01-eval-design`**: If you want a properly stratified dataset beyond the v0 30 inputs.
- **`02-metric-design`**: If you need multiple graders for different dimensions.