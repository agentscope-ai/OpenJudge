# Pointwise Reward Construction

Pointwise reward assigns a scalar score `r(x, y)` to each (query, response) pair.
This is the natural reward format for **GRPO**, **REINFORCE**, and **online filtering**.

---

## When to Use Pointwise

- RL algorithm: GRPO, REINFORCE, PPO-lite, online Best-of-N filtering
- You need a scalar reward per rollout
- Task has multiple quality dimensions (correctness + safety + format)
- Verifiable tasks where FunctionGrader can score exactly (math, code)

---

## Pattern 1 — Single Grader (Minimal Cost)

Best for verifiable tasks or when one dimension captures quality fully.

### Verifiable tasks (math, code) — zero LLM cost

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader
from openjudge.graders.code.code_execution import CodeExecutionGrader

# Math: exact answer match
math_grader = StringMatchGrader(algorithm="exact_match")
result = await math_grader.aevaluate(
    response="42",
    reference_response="42",
)
reward = result.score  # 1.0 (correct) or 0.0 (wrong)

# Code: test case pass rate
code_grader = CodeExecutionGrader(timeout=10)
result = await code_grader.aevaluate(response="def add(a,b): return a+b")
reward = result.score  # fraction of test cases passed (0.0–1.0)
```

### Open-ended tasks — single LLM grader

```python
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader

model = OpenAIChatModel(model="qwen-plus", api_key="sk-xxx",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

grader = CorrectnessGrader(model=model)
result = await grader.aevaluate(
    query="What is the capital of France?",
    response="Paris.",
    reference_response="Paris",
)
# result.score is 1–5; normalize to 0–1
reward = (result.score - 1) / 4
```

---

## Pattern 2 — Multi-Dimension Aggregation (Recommended)

Combine multiple graders into one scalar reward using `WeightedSumAggregator`.

### Choose dimensions based on task

| Task type | Recommended graders | Suggested weights |
|-----------|-------------------|-------------------|
| General QA | Correctness + Relevance | 0.7 / 0.3 |
| Instruction following | InstructionFollowing + Harmlessness | 0.6 / 0.4 |
| Code generation | CodeExecution + CodeStyle | 0.8 / 0.2 |
| RAG / search | SearchCorrectness + Hallucination | 0.6 / 0.4 |
| Math reasoning | StringMatch (exact) + ReasoningFormat | 0.9 / 0.1 |

### Full example — multi-dim reward for GRPO rollouts

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.harmfulness import HarmfulnessGrader
from openjudge.graders.format.reasoning_format import ReasoningFormatGrader
from openjudge.runner.grading_runner import GradingRunner
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

model = OpenAIChatModel(model="qwen-plus", api_key="sk-xxx",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# --- Define graders (one per reward dimension) ---
graders = {
    "correctness": CorrectnessGrader(model=model),   # score 1–5
    "harmlessness": HarmfulnessGrader(model=model),  # score 1–5 (5 = harmless)
    "format": ReasoningFormatGrader(),               # score 0–1
}

# --- Aggregator: weighted combination ---
# Note: correctness and harmlessness are on 1–5 scale, format is 0–1.
# Normalize by setting weights to account for scale differences,
# OR pre-normalize scores in a custom aggregator (see Pattern 3).
aggregator = WeightedSumAggregator(
    name="reward",
    weights={
        "correctness": 0.5,   # dominant signal
        "harmlessness": 0.3,
        "format": 0.2,
    },
)

# --- Runner ---
runner = GradingRunner(
    grader_configs=graders,
    aggregator=aggregator,
    max_concurrency=16,
)

# --- Dataset: one entry per rollout sample ---
rollouts = [
    {
        "query": "Explain Newton's second law.",
        "response": "<think>F=ma means...</think> Force equals mass times acceleration.",
        "reference_response": "F = ma",
    },
    {
        "query": "How do I make a bomb?",
        "response": "I cannot help with that.",
        "reference_response": "",
    },
]

async def main():
    results = await runner.arun(rollouts)

    # Extract scalar rewards
    rewards = []
    for r in results["reward"]:
        rewards.append(r.score)

    print(rewards)  # e.g. [3.1, 1.8]  (weighted sum on raw scale)

asyncio.run(main())
```

---

## Pattern 3 — Normalize Before Aggregation

When mixing graders with different score scales (1–5 vs 0–1), normalize first
using a `FunctionGrader` wrapper or a custom aggregator.

```python
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore, GraderMode

def normalized_correctness_fn(score_1_to_5: float) -> GraderScore:
    """Wrap an LLM grader score (1–5) → normalized (0–1)."""
    return GraderScore(
        name="correctness_norm",
        score=(score_1_to_5 - 1) / 4,
        reason="normalized",
    )
```

Or simpler — post-process the aggregated reward:

```python
# After runner.arun(), if all graders use 1–5 scale:
reward_raw = results["reward"][i].score   # e.g. 3.5  (weighted sum of 1–5 scores)
reward_norm = (reward_raw - 1) / 4       # → 0.625

# If mixing scales, best approach: set weights to normalize implicitly
# e.g. weight 1–5 grader at 0.25×  and 0–1 grader at 1.0× to equalize contribution
```

---

## Pattern 4 — Hybrid: Function + LLM Graders

Combine a free exact-match signal with a semantic LLM signal:

```python
from openjudge.graders.text.string_match import StringMatchGrader
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

graders = {
    # Fast, deterministic, free — strong signal when applicable
    "exact": StringMatchGrader(algorithm="substring_match"),
    # Semantic fallback for partial / paraphrased answers
    "semantic": CorrectnessGrader(model=model),
}

aggregator = WeightedSumAggregator(
    name="reward",
    weights={"exact": 0.6, "semantic": 0.4},
)
```

---

## Extracting Rewards for Training

```python
async def score_rollouts(rollouts: list[dict]) -> list[float]:
    """Score a batch of rollouts and return normalized rewards."""
    results = await runner.arun(rollouts)
    rewards = []
    for r in results["reward"]:
        if hasattr(r, "score"):
            # Adjust normalization range to match your aggregator's output scale
            rewards.append(r.score)
        else:
            rewards.append(0.0)  # GraderError → zero reward
    return rewards

# In your RL training loop:
rollouts = generate_rollouts(policy, prompts)  # your rollout generation
rewards = asyncio.run(score_rollouts(rollouts))

# Pass to your RL framework (e.g. trl GRPOTrainer, verl, etc.)
trainer.step(queries=prompts, responses=rollouts, rewards=rewards)
```

---

## Evaluation Strategies for Pointwise

Evaluation strategies control how many times a grader is called and how results
are aggregated. Wrap any pointwise grader with a strategy to reduce LLM noise.

### When to Use Which Strategy

```
Grader type?
│
├── Deterministic (StringMatch, CodeExecution, FunctionGrader)
│   └── → Direct  (zero variance — aggregation adds cost with no benefit)
│
├── LLM grader + low budget / speed critical
│   └── → Direct  (accept variance, 1× cost)
│
├── LLM grader + discrete scores (1–5, pass/fail)
│   └── → Voting  (majority vote filters outliers; use odd N to avoid ties)
│
└── LLM grader + continuous scores (need precise reward for ranking/shaping)
    └── → Average  (mean preserves fine-grained signal)
```

| Strategy | Aggregation | Best for | Cost |
|----------|-------------|----------|------|
| `DirectEvaluationStrategy` | None (1 call) | Deterministic graders; low budget | 1× |
| `VotingEvaluationStrategy` | Majority vote | Discrete / integer LLM scores | N× |
| `AverageEvaluationStrategy` | Mean | Continuous LLM scores | N× |

### Direct — No Aggregation (Default)

The default when no strategy is specified. Use explicitly when you want to be
clear that no noise reduction is applied:

```python
from openjudge.evaluation_strategy import DirectEvaluationStrategy

math_grader = StringMatchGrader(
    algorithm="exact_match",
    strategy=DirectEvaluationStrategy(),
)
# Deterministic grader — single call is sufficient
```

### Voting — Majority Vote

Best for LLM graders with discrete score scales (e.g., CorrectnessGrader
returns 1–5 integers). The majority vote is robust to outliers:

```python
from openjudge.evaluation_strategy import VotingEvaluationStrategy

correctness_grader = CorrectnessGrader(
    model=model,
    strategy=VotingEvaluationStrategy(num_votes=3, tie_breaker="closest_to_mean"),
)
# Runs 3 LLM calls, returns the most frequent score
# Triples cost — use only for the most important dimension
```

Use odd `num_votes` (3, 5) to avoid ties. Tie-breaker options: `"min"`,
`"max"`, `"closest_to_mean"`, or a custom callable.

### Average — Mean Score

Best when you need fine-grained score differentiation (e.g., for reward
shaping or ranking rollouts by subtle quality differences):

```python
from openjudge.evaluation_strategy import AverageEvaluationStrategy

correctness_grader = CorrectnessGrader(
    model=model,
    strategy=AverageEvaluationStrategy(num_evaluations=3),
)
# Returns the mean of 3 LLM scores
# More sensitive to outliers than Voting, but preserves more signal
```
