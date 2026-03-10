---
name: openjudge
description: >
  Build custom LLM evaluation pipelines using the OpenJudge framework.
  Covers selecting and configuring graders (LLM-based, function-based, agentic),
  running batch evaluations with GradingRunner, combining scores with aggregators,
  applying evaluation strategies (voting, average), auto-generating graders from
  data, and analyzing results (pairwise win rates, statistics, validation metrics).
  Use when the user wants to evaluate LLM outputs, compare multiple models,
  design scoring criteria, or build an automated evaluation system.
---

# OpenJudge Skill

Build evaluation pipelines for LLM applications using the `openjudge` library.

## When to Use This Skill

- User wants to evaluate LLM output quality (correctness, relevance, hallucination, etc.)
- User wants to compare two or more models and rank them
- User wants to design a scoring rubric and automate evaluation
- User wants to analyze evaluation results statistically
- User wants to build a reward model or quality filter

## Sub-documents — Read When Relevant

| Topic | File | Read when… |
|-------|------|------------|
| Grader selection & configuration | `graders.md` | User needs to pick or configure an evaluator |
| Batch evaluation pipeline | `pipeline.md` | User needs to run evaluation over a dataset |
| Auto-generate graders from data | `generator.md` | No rubric yet; generate from labeled examples |
| Analyze & compare results | `analyzer.md` | User wants win rates, statistics, or metrics |

Read the relevant sub-document **before** writing any code.

## Install

```bash
pip install py-openjudge
```

## Architecture Overview

```
Dataset (List[dict])
    │
    ▼
GradingRunner                    ← orchestrates everything
    │
    ├─► Grader A ──► EvaluationStrategy ──► _aevaluate() ──► GraderScore / GraderRank
    ├─► Grader B ──► EvaluationStrategy ──► _aevaluate() ──► GraderScore / GraderRank
    └─► Grader C ...
    │
    ├─► Aggregator (optional)    ← combine multiple grader scores into one
    │
    └─► RunnerResult             ← {grader_name: [GraderScore, ...]}
            │
            ▼
        Analyzer                 ← statistics, win rates, validation metrics
```

## 5-Minute Quick Start

Evaluate responses for correctness using a built-in grader:

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.runner.grading_runner import GradingRunner

# 1. Configure the judge model (OpenAI-compatible endpoint)
model = OpenAIChatModel(
    model="qwen-plus",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. Instantiate a grader
grader = CorrectnessGrader(model=model)

# 3. Prepare dataset
dataset = [
    {
        "query": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "reference_response": "Paris.",
    },
    {
        "query": "What is 2 + 2?",
        "response": "The answer is five.",
        "reference_response": "4.",
    },
]

# 4. Run evaluation
async def main():
    runner = GradingRunner(
        grader_configs={"correctness": grader},
        max_concurrency=8,
    )
    results = await runner.arun(dataset)

    for i, result in enumerate(results["correctness"]):
        print(f"[{i}] score={result.score}  reason={result.reason}")

asyncio.run(main())
```

**Expected output:**
```
[0] score=5  reason=The response accurately states Paris as capital...
[1] score=1  reason=The response gives the wrong answer (five vs 4)...
```

## Key Data Types

| Type | Description |
|------|-------------|
| `GraderScore` | Pointwise result: `.score` (float), `.reason` (str), `.metadata` (dict) |
| `GraderRank` | Listwise result: `.rank` (List[int]), `.reason` (str), `.metadata` (dict) |
| `GraderError` | Error during evaluation: `.error` (str), `.reason` (str) |
| `RunnerResult` | `Dict[str, List[GraderResult]]` — keyed by grader name |

## Result Handling Pattern

```python
from openjudge.graders.schema import GraderScore, GraderRank, GraderError

for grader_name, grader_results in results.items():
    for i, result in enumerate(grader_results):
        if isinstance(result, GraderScore):
            print(f"{grader_name}[{i}]: score={result.score}")
        elif isinstance(result, GraderRank):
            print(f"{grader_name}[{i}]: rank={result.rank}")
        elif isinstance(result, GraderError):
            print(f"{grader_name}[{i}]: ERROR — {result.error}")
```

## Model Configuration

All LLM-based graders accept either a `BaseChatModel` instance or a dict config:

```python
# Option A: instance
from openjudge.models.openai_chat_model import OpenAIChatModel
model = OpenAIChatModel(model="gpt-4o", api_key="sk-...")

# Option B: dict (auto-creates OpenAIChatModel)
model_cfg = {"model": "gpt-4o", "api_key": "sk-..."}
grader = CorrectnessGrader(model=model_cfg)

# OpenAI-compatible endpoints (DashScope / local / etc.)
model = OpenAIChatModel(
    model="qwen-plus",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```
