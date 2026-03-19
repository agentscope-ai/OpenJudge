---
name: rl-reward
description: >
  Build RL reward signals using the OpenJudge framework.
  Covers choosing between pointwise and pairwise reward strategies based on
  RL algorithm, task type, and cost; aggregating multi-dimensional pointwise
  scores into a scalar reward; pairwise tournament reward for GRPO on subjective
  tasks (net win rate across group rollouts); generating preference pairs for
  DPO/RLAIF; and normalizing scores for training stability.
  Use when building reward models, scoring rollouts for GRPO/REINFORCE,
  generating preference data for DPO, or doing Best-of-N selection.
---

# RL Reward Construction with OpenJudge

Build reward signals for reinforcement learning from human feedback (RLHF) and
reinforcement learning from AI feedback (RLAIF) using the `openjudge` library.

## When to Use This Skill

- Building scalar rewards for GRPO / REINFORCE rollout scoring
- Generating (chosen, rejected) preference pairs for DPO / IPO
- Best-of-N candidate selection
- Multi-dimensional reward shaping (correctness + safety + format)
- Replacing or bootstrapping a reward model with LLM-as-judge

## Step 1 — Choose Your Reward Strategy

Use this decision tree **before** writing any code:

```
RL Algorithm + Task type?
│
├── GRPO / REINFORCE — Verifiable task (math, code, structured output)
│   └── → POINTWISE  ✅  (FunctionGrader, exact score, zero LLM cost)
│
├── GRPO / REINFORCE — Subjective task (instruction following, dialogue, summarization)
│   └── → PAIRWISE TOURNAMENT  ✅  (compare each rollout vs all others in group,
│                                    reward = net win rate within group)
│
├── DPO / IPO / SLiC — need (chosen, rejected) pairs
│   └── → PAIRWISE  ✅  (two-way comparison, return winner/loser)
│
└── Best-of-N / reranking — rank N candidates
    └── → LISTWISE  ✅  (single call ranks all N at once)
```

```
Cost constraint?
├── Low budget
│   └── FunctionGrader (free) → pointwise; or pairwise with small judge model
│
├── Medium budget
│   └── Pointwise: 2–3 LLM graders + WeightedSumAggregator
│   └── Pairwise tournament: 1 LLM judge, N*(N-1)/2 comparisons per group
│
└── High quality / no cost limit
    └── Pointwise voting (3–5 calls) or pairwise with strong judge + debiasing
```

## Sub-documents — Read When Relevant

| Topic | File | Read when… |
|-------|------|------------|
| Pointwise multi-dim reward | `pointwise.md` | GRPO on verifiable tasks; multi-dimension scoring |
| Pairwise reward | `pairwise.md` | GRPO on subjective tasks (tournament); DPO/RLAIF preference pairs |

Read the relevant sub-document **before** writing any code.

## Install

```bash
pip install py-openjudge
```

## Strategy Comparison

| Strategy | Output | Reward signal | Typical use | Cost |
|----------|--------|---------------|-------------|------|
| **Pointwise** | scalar per response | direct reward `r(x, y)` | GRPO on verifiable tasks, filtering | Low–Medium |
| **Pairwise Tournament** | net win rate per response | relative reward within group | GRPO on subjective tasks | Medium (N²/2 calls) |
| **Pairwise** | winner/loser pair | implicit preference `y+ > y-` | DPO, IPO, RLAIF preference data | Medium |
| **Listwise** | rank over N responses | ordinal reward / reranking | Best-of-N, reranking | Medium–High |

## Score Normalization

All graders return scores on different scales. **Always normalize** before feeding into RL:

```python
def normalize(score: float, min_score: float, max_score: float) -> float:
    """Map [min_score, max_score] → [0.0, 1.0]."""
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)

# LLM graders (common/*) return 1–5 → normalize to 0–1
reward = normalize(result.score, min_score=1, max_score=5)

# FunctionGrader / text graders already return 0–1 → no normalization needed
```

## Evaluation Strategies

Evaluation strategies control **how many times** a grader is called and **how
results are aggregated**. They are independent of the grader itself.

### Choose Your Strategy

```
Grader type?
│
├── Deterministic (FunctionGrader, StringMatch, CodeExecution, etc.)
│   └── → Direct  (zero variance, no need for aggregation)
│
├── LLM grader — Pointwise scoring
│   │
│   ├── Budget limited / speed critical
│   │   └── → Direct  (accept variance, 1× cost)
│   │
│   ├── Discrete scores (1–5 integer, pass/fail, binary)
│   │   └── → Voting  (majority vote, robust to outliers, N× cost)
│   │
│   └── Continuous / fine-grained scores (need precise ranking)
│       └── → Average  (mean, preserves signal, N× cost)
│
└── LLM grader — Pairwise GRPO tournament
    └── → GRPOTournament  (all-pairs comparison, net win rate)
```

| Strategy | Aggregation | Best for | Cost |
|----------|-------------|----------|------|
| `DirectEvaluationStrategy` | None | Deterministic graders; low budget | 1× |
| `VotingEvaluationStrategy` | Majority vote | Discrete / integer LLM scores | N× |
| `AverageEvaluationStrategy` | Mean | Continuous LLM scores | N× |
| `GRPOTournamentEvaluationStrategy` | Net win rate | Pairwise GRPO on subjective tasks | N²/2× |

All strategies are imported from `openjudge.evaluation_strategy`.

### Pointwise — Noise Reduction with Voting / Average

For high-variance LLM judges, wrap any grader with `VotingEvaluationStrategy`
to run N calls and take the majority vote:

```python
from openjudge.evaluation_strategy import VotingEvaluationStrategy

grader = CorrectnessGrader(
    model=model,
    strategy=VotingEvaluationStrategy(num_votes=3, tie_breaker="closest_to_mean"),
)
# Now each call internally runs 3 LLM evaluations and returns the most common score
```

Use odd `num_votes` (3, 5) to avoid ties.

### Pairwise — GRPO Tournament

For GRPO on subjective tasks, use `GRPOTournamentEvaluationStrategy` to run
all-pairs comparison and compute net win rate per rollout:

```python
from openjudge.evaluation_strategy import GRPOTournamentEvaluationStrategy

strategy = GRPOTournamentEvaluationStrategy(debiased=False)
results = await strategy.execute(
    pairwise_grader.aevaluate,
    query="Write a haiku about the ocean.",
    responses=["rollout_1", "rollout_2", "rollout_3", "rollout_4"],
)
rewards = [r.score for r in results]  # net win rates in [-1.0, 1.0]
```

Set `debiased=True` to run each pair in both orders and only count consistent
results (doubles LLM calls but mitigates position bias).
