# Analyzer Reference

Analyzers process `RunnerResult` to produce aggregated insights:
statistics, pairwise rankings, and validation metrics against ground truth.

All analyzers follow the same interface:
```python
result = analyzer.analyze(dataset, grader_results, **kwargs)
```

---

## PairwiseAnalyzer — Model Comparison & Win Rates

Use when evaluating multiple models head-to-head.
Computes win rates, a win matrix, and final rankings.

### Setup

Dataset samples must contain a `metadata` dict with `model_a` and `model_b` keys:

```python
dataset = [
    {"metadata": {"model_a": "gpt-4o", "model_b": "qwen-max"}},
    {"metadata": {"model_a": "qwen-max", "model_b": "gpt-4o"}},  # swapped pair
    ...
]
```

Grader results use score conventions:
- `score >= 0.5` → `model_a` wins
- `score < 0.5` → `model_b` wins

### Example

```python
from openjudge.analyzer.pairwise_analyzer import PairwiseAnalyzer
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.runner.grading_runner import GradingRunner

# Build a pairwise judge grader
judge = LLMGrader(
    model=model,
    name="pairwise_judge",
    mode=GraderMode.POINTWISE,
    template="""
You are a judge. Compare Response A and Response B for the given query.
Score 1.0 if Response A is better, 0.0 if Response B is better, 0.5 if tied.

Query: {query}
Response A: {response_a}
Response B: {response_b}

JSON: {{"score": <float>, "reason": "<explanation>"}}
""",
)

# Dataset: pairwise samples (typically generated with position swap for bias correction)
dataset = [
    {
        "query": "What is quantum computing?",
        "response_a": "GPT-4o answer...",
        "response_b": "Qwen-max answer...",
        "metadata": {"model_a": "gpt-4o", "model_b": "qwen-max"},
    },
    {
        "query": "What is quantum computing?",
        "response_a": "Qwen-max answer...",
        "response_b": "GPT-4o answer...",
        "metadata": {"model_a": "qwen-max", "model_b": "gpt-4o"},  # swapped
    },
]

runner = GradingRunner(grader_configs={"judge": judge}, max_concurrency=8)
results = await runner.arun(dataset)

# Analyze
analyzer = PairwiseAnalyzer(model_names=["gpt-4o", "qwen-max"])
analysis = analyzer.analyze(dataset, results["judge"])

print(f"Best model: {analysis.best_model}")
print(f"Rankings:   {analysis.rankings}")
print(f"Win rates:  {analysis.win_rates}")
print(f"Win matrix: {analysis.win_matrix}")
```

**Result fields:**

| Field | Type | Description |
|-------|------|-------------|
| `best_model` | str | Model with highest win rate |
| `worst_model` | str | Model with lowest win rate |
| `win_rates` | `Dict[str, float]` | Win rate per model (0.0–1.0) |
| `rankings` | `List[Tuple[str, float]]` | Sorted by win rate descending |
| `win_matrix` | `Dict[str, Dict[str, float]]` | `win_matrix[A][B]` = how often A beats B |
| `total_comparisons` | int | Total pairwise samples analyzed |

---

## Statistical Analyzers

### DistributionAnalyzer

Computes score distribution statistics for a single grader's results.

```python
from openjudge.analyzer.statistical.distribution_analyzer import DistributionAnalyzer

analyzer = DistributionAnalyzer()
result = analyzer.analyze(dataset, results["correctness"])

print(f"mean={result.mean:.3f}")
print(f"median={result.median:.3f}")
print(f"stdev={result.stdev:.3f}")
print(f"min={result.min_score}  max={result.max_score}")
```

**Result fields:** `mean`, `median`, `stdev`, `min_score`, `max_score`

---

### ConsistencyAnalyzer

Measures how consistent a grader is across two independent runs on the same samples.
Returns Pearson correlation between the two score lists.

```python
from openjudge.analyzer.statistical.consistency_analyzer import ConsistencyAnalyzer

# Run the same grader twice
runner = GradingRunner(grader_configs={"correctness": grader}, max_concurrency=8)
run1 = await runner.arun(dataset)
run2 = await runner.arun(dataset)

analyzer = ConsistencyAnalyzer()
result = analyzer.analyze(
    dataset=dataset,
    grader_results=run1["correctness"],
    another_grader_results=run2["correctness"],
)

print(f"Consistency (Pearson r): {result.consistency:.4f}")
# 1.0 = perfectly consistent; 0.0 = no correlation
```

**Result fields:** `consistency` (float, Pearson r)

---

## Validation Analyzers

Validation analyzers compare grader scores against **ground truth labels** in the dataset.

**Prerequisite:** Each sample in `dataset` must have a label field (default key: `"label"`).

```python
dataset = [
    {"query": "...", "response": "...", "label": 1},   # ground truth: correct
    {"query": "...", "response": "...", "label": 0},   # ground truth: incorrect
]
```

### AccuracyAnalyzer

Fraction of samples where `grader.score == label`.

```python
from openjudge.analyzer.validation import AccuracyAnalyzer

analyzer = AccuracyAnalyzer()
result = analyzer.analyze(dataset, grader_results, label_path="label")
print(f"Accuracy: {result.accuracy:.2%}")
```

### F1ScoreAnalyzer

Harmonic mean of precision and recall.

```python
from openjudge.analyzer.validation import F1ScoreAnalyzer

analyzer = F1ScoreAnalyzer()
result = analyzer.analyze(dataset, grader_results, label_path="label")
print(f"F1: {result.f1_score:.4f}")
```

### PrecisionAnalyzer / RecallAnalyzer

```python
from openjudge.analyzer.validation import PrecisionAnalyzer, RecallAnalyzer

precision_result = PrecisionAnalyzer().analyze(dataset, grader_results)
recall_result    = RecallAnalyzer().analyze(dataset, grader_results)
print(f"Precision: {precision_result.precision:.4f}")
print(f"Recall:    {recall_result.recall:.4f}")
```

### FalsePositiveAnalyzer / FalseNegativeAnalyzer

```python
from openjudge.analyzer.validation import FalsePositiveAnalyzer, FalseNegativeAnalyzer

fp_result = FalsePositiveAnalyzer().analyze(dataset, grader_results)
fn_result = FalseNegativeAnalyzer().analyze(dataset, grader_results)
print(f"False positive rate: {fp_result.false_positive_rate:.4f}")
print(f"False negative rate: {fn_result.false_negative_rate:.4f}")
```

### CorrelationAnalyzer

Pearson/Spearman correlation between grader scores and numeric labels.

```python
from openjudge.analyzer.validation import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
result = analyzer.analyze(dataset, grader_results, label_path="score_label")
print(f"Pearson r:  {result.pearson_correlation:.4f}")
print(f"Spearman r: {result.spearman_correlation:.4f}")
```

---

## All Validation Analyzers — Summary Table

| Analyzer | Key result field | Use when |
|----------|-----------------|----------|
| `AccuracyAnalyzer` | `.accuracy` | Binary or categorical grader vs label |
| `F1ScoreAnalyzer` | `.f1_score` | Binary classification, imbalanced labels |
| `PrecisionAnalyzer` | `.precision` | Cost of false positives is high |
| `RecallAnalyzer` | `.recall` | Cost of false negatives is high |
| `FalsePositiveAnalyzer` | `.false_positive_rate` | Measure over-flagging |
| `FalseNegativeAnalyzer` | `.false_negative_rate` | Measure under-detection |
| `CorrelationAnalyzer` | `.pearson_correlation`, `.spearman_correlation` | Continuous score calibration |

---

## Complete Analysis Workflow

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.runner.grading_runner import GradingRunner
from openjudge.analyzer.statistical.distribution_analyzer import DistributionAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer, F1ScoreAnalyzer

model = OpenAIChatModel(model="qwen-plus", api_key="sk-xxx",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# Dataset with ground truth labels
dataset = [
    {"query": "2+2?", "response": "4",    "reference_response": "4", "label": 1},
    {"query": "2+2?", "response": "Five", "reference_response": "4", "label": 0},
    {"query": "Capital of France?", "response": "Paris", "reference_response": "Paris", "label": 1},
    {"query": "Capital of France?", "response": "London", "reference_response": "Paris", "label": 0},
]

async def main():
    runner = GradingRunner(
        grader_configs={"correctness": CorrectnessGrader(model=model)},
        max_concurrency=4,
    )
    results = await runner.arun(dataset)
    grader_results = results["correctness"]

    # Score distribution
    dist = DistributionAnalyzer().analyze(dataset, grader_results)
    print(f"Score distribution: mean={dist.mean:.2f}, stdev={dist.stdev:.2f}")

    # Validation against labels (binarize: score >= 3 → correct)
    binary_results = []
    from openjudge.graders.schema import GraderScore
    for r in grader_results:
        if isinstance(r, GraderScore):
            binary_results.append(GraderScore(
                name=r.name, score=1.0 if r.score >= 3 else 0.0, reason=r.reason
            ))

    acc = AccuracyAnalyzer().analyze(dataset, binary_results, label_path="label")
    f1  = F1ScoreAnalyzer().analyze(dataset, binary_results, label_path="label")
    print(f"Accuracy: {acc.accuracy:.2%}")
    print(f"F1 Score: {f1.f1_score:.4f}")

asyncio.run(main())
```
