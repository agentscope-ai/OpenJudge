# Generator Reference

Generators automatically create `LLMGrader` instances by deriving evaluation rubrics
from data — no manual rubric writing required.

**Use a generator when:**
- You have labeled examples (query + response + score/rank) but no rubric
- You want to adapt evaluation criteria to a specific task domain
- You need to bootstrap a grader from scratch

---

## Two Generator Types

| Generator | Input | Best for |
|-----------|-------|----------|
| `SimpleRubricsGenerator` | Task description + optional sample queries | Cold start, no labeled data needed |
| `IterativeRubricsGenerator` | Labeled dataset (query + response + score) | Better quality, learns from preference data |

Both return a ready-to-use `LLMGrader`.

---

## SimpleRubricsGenerator

Generates rubrics from a **task description** and optional sample queries.
No labeled data required — fastest way to bootstrap a grader.

### Config parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grader_name` | str | `"Generated Grader"` | Name for the generated grader |
| `model` | BaseChatModel | required | LLM used to generate rubrics |
| `task_description` | str | `""` | What the task is about |
| `scenario` | str | None | Usage context (e.g., "customer support chatbot") |
| `grader_mode` | GraderMode | `POINTWISE` | `POINTWISE` or `LISTWISE` |
| `language` | LanguageEnum | `EN` | `EN` or `ZH` |
| `min_score` | int | `0` | Min score (pointwise mode) |
| `max_score` | int | `1` | Max score (pointwise mode) |

### Example — pointwise grader from task description

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.generator.simple_rubric.generator import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
)

model = OpenAIChatModel(model="qwen-plus", api_key="sk-xxx",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

config = SimpleRubricsGeneratorConfig(
    grader_name="Customer Support Grader",
    model=model,
    task_description="Customer support chatbot for an e-commerce platform",
    scenario="Customers asking about orders, returns, and shipping",
    min_score=0,
    max_score=1,
)

generator = SimpleRubricsGenerator(config)

async def main():
    # Option A: pass sample queries explicitly
    grader = await generator.generate(
        dataset=[],
        sample_queries=[
            "Where is my order?",
            "How do I return a product?",
            "What is the shipping time?",
        ],
    )

    # Option B: extract queries from dataset automatically (uses first 5)
    dataset = [{"query": "Where is my order?", "response": "..."}]
    grader = await generator.generate(dataset=dataset)

    # Use the generated grader
    result = await grader.aevaluate(
        query="How do I cancel my order?",
        response="You can cancel your order within 24 hours from the order page.",
    )
    print(f"score={result.score}  reason={result.reason}")

asyncio.run(main())
```

### Example — listwise (ranking) grader

```python
from openjudge.graders.schema import GraderMode

config = SimpleRubricsGeneratorConfig(
    grader_name="Response Ranker",
    model=model,
    task_description="Compare and rank responses to customer questions",
    grader_mode=GraderMode.LISTWISE,
)
generator = SimpleRubricsGenerator(config)
grader = await generator.generate(dataset=[])
```

---

## IterativeRubricsGenerator

Derives rubrics from **labeled preference data** using an iterative Propose-Evaluate-Revise loop,
then selects an optimal non-redundant subset via information-theoretic MCR² selection.

Based on the paper: *Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling*

### Two config classes (choose based on mode)

**Pointwise:**
```python
from openjudge.generator.iterative_rubric.generator import (
    IterativeRubricsGenerator,
    IterativePointwiseRubricsGeneratorConfig,
)

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="My Pointwise Grader",
    model=model,
    min_score=0,
    max_score=1,
    # optional tuning:
    task_description="Evaluate answers to science questions",
    enable_categorization=False,
    max_epochs=3,
    batch_size=10,
)
```

**Listwise:**
```python
from openjudge.generator.iterative_rubric.generator import (
    IterativeRubricsGenerator,
    IterativeListwiseRubricsGeneratorConfig,
)

config = IterativeListwiseRubricsGeneratorConfig(
    grader_name="My Listwise Grader",
    model=model,
)
```

### Dataset format

**Pointwise dataset** — each sample needs `query`, `response`, and optionally `label_score` (for validation):

```python
pointwise_dataset = [
    {"query": "What causes rain?", "response": "Water vapour condenses...", "label_score": 1},
    {"query": "What is DNA?",      "response": "DNA is a molecule...",       "label_score": 1},
    {"query": "What is DNA?",      "response": "I don't know.",              "label_score": 0},
]
```

**Listwise dataset** — each sample needs `query`, `responses` list, and optionally `label_rank` (for validation):

```python
listwise_dataset = [
    {
        "query": "Explain photosynthesis",
        "responses": [
            "Plants use sunlight, CO₂, and water to produce glucose.",
            "Plants need sunlight.",
        ],
        "label_rank": [1, 2],   # 1 = best
    },
]
```

### Full example

```python
import asyncio
from openjudge.generator.iterative_rubric.generator import (
    IterativeRubricsGenerator,
    IterativePointwiseRubricsGeneratorConfig,
)

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="Science QA Grader",
    model=model,
    task_description="Evaluate factual answers to science questions",
    min_score=0,
    max_score=1,
    max_epochs=3,
    batch_size=5,
)

generator = IterativeRubricsGenerator(config)

async def main():
    train_data = [
        {"query": "What is gravity?",  "response": "A force attracting masses.", "label_score": 1},
        {"query": "What is gravity?",  "response": "Something heavy.",           "label_score": 0},
        {"query": "What is entropy?",  "response": "Measure of disorder.",       "label_score": 1},
        {"query": "What is entropy?",  "response": "A type of energy.",          "label_score": 0},
    ]

    # Generate grader — may take several minutes for large datasets
    grader = await generator.generate(dataset=train_data)

    # Evaluate new samples
    result = await grader.aevaluate(
        query="What is osmosis?",
        response="Osmosis is the movement of water across a semi-permeable membrane.",
    )
    print(f"score={result.score}  reason={result.reason}")

asyncio.run(main())
```

### Key config parameters (IterativeRubricsGeneratorConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_categorization` | `False` | Merge similar rubrics via LLM (slower, more organised) |
| `categories_number` | `5` | Target category count (only when categorization enabled) |
| `max_epochs` | `5` | Max Propose-Evaluate-Revise iterations per sample |
| `batch_size` | `10` | Samples per batch |
| `max_total_rubrics` | `200` | Cap on total rubrics collected |
| `min_increment_threshold` | `0.002` | Convergence threshold for MCR² selection |
| `patience` | `2` | Consecutive low-increment batches before early stop |

**Sampling mode is auto-selected:**
- `≤ 100 samples` → all_samples mode (process all concurrently)
- `> 100 samples` → smart_sampling mode (MCR²-guided batch iteration)

---

## Using a Generated Grader in GradingRunner

The returned `LLMGrader` is a standard grader — plug it directly into a runner:

```python
from openjudge.runner.grading_runner import GradingRunner

grader = await generator.generate(dataset=train_data)

runner = GradingRunner(
    grader_configs={"auto_rubric": grader},
    max_concurrency=8,
)
test_dataset = [{"query": "...", "response": "..."}]
results = await runner.arun(test_dataset)
```
