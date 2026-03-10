# Pipeline Reference

The pipeline layer handles batch evaluation: running graders over datasets,
controlling concurrency, combining multiple grader scores, and stabilizing
noisy LLM evaluations.

---

## GradingRunner

`GradingRunner` is the main entry point for batch evaluation.
It runs all configured graders over a dataset concurrently.

### Constructor

```python
from openjudge.runner.grading_runner import GradingRunner, GraderConfig

runner = GradingRunner(
    grader_configs,      # Dict[str, grader | (grader, mapper) | GraderConfig]
    max_concurrency=32,  # max parallel API calls
    aggregators=None,    # optional aggregator(s)
    show_progress=True,  # tqdm progress bar
    executor=None,       # custom resource executor (rarely needed)
)
```

### Running evaluation

```python
# Single dataset
results = await runner.arun(dataset)          # RunnerResult

# Multiple datasets (shared concurrency pool)
all_results = await runner.arun_multiple_datasets([dataset_a, dataset_b])
```

### Result structure

```
RunnerResult = Dict[str, List[GraderResult]]

{
    "grader_a": [GraderScore(...), GraderScore(...), GraderError(...)],
    "grader_b": [GraderScore(...), GraderScore(...), GraderScore(...)],
}
```

Each list is indexed the same as the input `dataset` list.

---

## GraderConfig — Input Formats

`grader_configs` accepts four equivalent formats:

```python
from openjudge.runner.grading_runner import GraderConfig

# Format 1: bare grader instance (most common)
configs = {"correctness": CorrectnessGrader(model=model)}

# Format 2: tuple (grader, mapper)
configs = {"correctness": (CorrectnessGrader(model=model), {"q": "query", "a": "response"})}

# Format 3: GraderConfig object
configs = {"correctness": GraderConfig(grader=CorrectnessGrader(model=model), mapper=...)}

# Format 4: dict
configs = {"correctness": {"grader": CorrectnessGrader(model=model), "mapper": None}}
```

---

## Mapper — Field Name Translation

Use a mapper when your dataset field names differ from what the grader expects.

### Dict mapper (field rename)

```python
# Dataset has "question" / "answer" but grader expects "query" / "response"
configs = {
    "correctness": GraderConfig(
        grader=CorrectnessGrader(model=model),
        mapper={"question": "query", "answer": "response"},
        #        dataset key → grader kwarg
    )
}
```

### Callable mapper (full transformation)

```python
def my_mapper(sample: dict) -> dict:
    return {
        "query": sample["input"],
        "response": sample["output"],
        "reference_response": sample.get("gold", ""),
        "context": " ".join(sample.get("docs", [])),
    }

configs = {
    "correctness": GraderConfig(grader=CorrectnessGrader(model=model), mapper=my_mapper)
}
```

---

## Multiple Graders in One Run

Run multiple graders over the same dataset in one pass:

```python
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.common.hallucination import HallucinationGrader

runner = GradingRunner(
    grader_configs={
        "correctness": CorrectnessGrader(model=model),
        "relevance":   RelevanceGrader(model=model),
        "hallucination": HallucinationGrader(model=model),
    },
    max_concurrency=16,
)

results = await runner.arun(dataset)
# results["correctness"][i], results["relevance"][i], results["hallucination"][i]
```

---

## WeightedSumAggregator — Combine Multiple Scores

Produce a single composite score from multiple graders per sample.

```python
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

aggregator = WeightedSumAggregator(
    name="overall",
    weights={
        "correctness":   0.5,
        "relevance":     0.3,
        "hallucination": 0.2,
    },
)

runner = GradingRunner(
    grader_configs={
        "correctness":   CorrectnessGrader(model=model),
        "relevance":     RelevanceGrader(model=model),
        "hallucination": HallucinationGrader(model=model),
    },
    aggregators=aggregator,
)

results = await runner.arun(dataset)
# results["overall"][i]  ← WeightedSumAggregator result (GraderScore)
# results["correctness"][i], results["relevance"][i], ...  ← individual scores
```

**Notes:**
- If `weights` is omitted, equal weights are used automatically.
- `GraderError` and `GraderRank` results are skipped in the weighted sum.
- Multiple aggregators can be passed as a list.

### Custom aggregator

```python
from openjudge.runner.aggregator.base_aggregator import BaseAggregator
from openjudge.graders.schema import GraderResult, GraderScore

class MinScoreAggregator(BaseAggregator):
    """Returns the minimum score across all graders."""

    def __call__(self, grader_results: dict[str, GraderResult], **kwargs) -> GraderResult:
        scores = [r.score for r in grader_results.values() if isinstance(r, GraderScore)]
        if not scores:
            return GraderScore(name=self.name, score=0.0, reason="No valid scores")
        return GraderScore(
            name=self.name,
            score=min(scores),
            reason=f"Min of {len(scores)} grader scores",
        )

aggregator = MinScoreAggregator(name="min_score")
```

---

## Evaluation Strategies — Reduce LLM Noise

Attach a strategy to any grader to call it multiple times and aggregate.

### VotingEvaluationStrategy

Run N times, return the most frequent score. Best for discrete scores (1–5).

```python
from openjudge.evaluation_strategy import VotingEvaluationStrategy, MIN

strategy = VotingEvaluationStrategy(
    num_votes=5,         # must be ≥ 2; odd numbers avoid ties
    tie_breaker=MIN,     # MIN | MAX | CLOSEST_TO_MEAN | custom callable
)

grader = CorrectnessGrader(model=model, strategy=strategy)
```

### AverageEvaluationStrategy

Run N times, return the mean score. Best for continuous scores.

```python
from openjudge.evaluation_strategy import AverageEvaluationStrategy

strategy = AverageEvaluationStrategy(num_evaluations=3)
grader = RelevanceGrader(model=model, strategy=strategy)
```

### DirectEvaluationStrategy (default)

Call once, return result as-is. This is the default when no strategy is specified.

```python
from openjudge.evaluation_strategy import DirectEvaluationStrategy

grader = CorrectnessGrader(model=model, strategy=DirectEvaluationStrategy())
```

---

## Concurrency Control

`max_concurrency` limits simultaneous LLM API calls across all graders and samples.

```python
runner = GradingRunner(
    grader_configs={"correctness": grader},
    max_concurrency=8,   # conservative for rate-limited APIs
)
```

The underlying `SemaphoreResourceExecutor` ensures the total number of in-flight
requests never exceeds `max_concurrency`, regardless of dataset size or number of graders.

---

## Complete Pipeline Example

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.text.string_match import StringMatchGrader
from openjudge.runner.grading_runner import GradingRunner, GraderConfig
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.evaluation_strategy import VotingEvaluationStrategy
from openjudge.graders.schema import GraderScore, GraderError

model = OpenAIChatModel(model="qwen-plus", api_key="sk-xxx",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# Voting strategy for LLM-based graders
voting = VotingEvaluationStrategy(num_votes=3)

dataset = [
    {
        "query": "What is the capital of France?",
        "response": "Paris",
        "reference": "Paris",
        "reference_response": "The capital of France is Paris.",
    },
]

runner = GradingRunner(
    grader_configs={
        "correctness": CorrectnessGrader(model=model, strategy=voting),
        "relevance":   RelevanceGrader(model=model, strategy=voting),
        "exact_match": GraderConfig(
            grader=StringMatchGrader(),
            mapper={"response": "response", "reference": "reference"},
        ),
    },
    aggregators=WeightedSumAggregator(
        name="overall",
        weights={"correctness": 0.5, "relevance": 0.3, "exact_match": 0.2},
    ),
    max_concurrency=8,
)

async def main():
    results = await runner.arun(dataset)
    for grader_name, grader_results in results.items():
        for i, result in enumerate(grader_results):
            if isinstance(result, GraderScore):
                print(f"[{grader_name}][{i}] score={result.score:.3f}")
            elif isinstance(result, GraderError):
                print(f"[{grader_name}][{i}] ERROR: {result.error}")

asyncio.run(main())
```
