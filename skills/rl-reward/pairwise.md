# Pairwise Reward Construction

Pairwise reward compares two responses to the same query and identifies which is
preferred. This covers two distinct use cases:

1. **GRPO Tournament** — score each rollout in a group by its net win rate against
   all other rollouts. Best for subjective tasks where absolute scoring is unreliable.
2. **Preference Pairs** — identify `y+ > y-` pairs for DPO / IPO / RLAIF.

OpenJudge implements pairwise comparison as a **LISTWISE grader with 2 candidates**
(`GraderMode.LISTWISE`). The output is a `GraderRank` where `rank[i] = 1` means
response `i+1` is the winner.

---

## When to Use Each Variant

| Variant | RL Algorithm | When |
|---------|-------------|------|
| **Tournament** | GRPO, REINFORCE | Subjective task; absolute scoring unreliable; want relative reward within group |
| **Preference Pairs** | DPO, IPO, SLiC, RLAIF | Need (chosen, rejected) dataset; offline preference labeling |

---

## Pattern 0 — GRPO Pairwise Tournament (Subjective Tasks)

For a GRPO group of N rollouts for the same prompt, compare every pair and assign
reward as **net win rate**: `r_i = (wins_i - losses_i) / (N - 1)`.

This reward is **relative within the group** (like standard GRPO normalization),
robust to absolute scale drift, and works well when LLM judges find relative
comparison easier than absolute scoring.

Use `GRPOTournamentEvaluationStrategy` — it handles all-pairs comparison,
concurrent execution, error skipping, and optional position-bias debiasing:

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.evaluation_strategy import GRPOTournamentEvaluationStrategy

model = OpenAIChatModel(
    model="qwen-plus",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

pairwise_grader = LLMGrader(
    model=model,
    name="preference",
    mode=GraderMode.LISTWISE,
    template="""Compare two responses to the query. Which one is better overall?
Consider accuracy, helpfulness, clarity, and safety.

Query: {query}
Response 1: {response_1}
Response 2: {response_2}

Respond in JSON: {{"rank": [<int>, <int>], "reason": "<brief explanation>"}}""",
)

strategy = GRPOTournamentEvaluationStrategy()

# --- Usage in GRPO training loop ---
async def main():
    query = "Write a haiku about the ocean."
    rollouts = [
        "Waves crash endlessly,\nSalt air fills the morning light,\nOcean breathes and sighs.",
        "The ocean is blue and big.",
        "Tides pull at the shore,\nDeep currents remember storms—\nsilence under waves.",
        "Water water everywhere,\nfish swim in the sea.",
    ]

    results = await strategy.execute(
        pairwise_grader.aevaluate,
        query=query,
        responses=rollouts,
    )
    rewards = [r.score for r in results]
    print(rewards)  # e.g. [0.67, -1.0, 1.0, -0.67]
    # Each result also has metadata: wins, losses, valid_comparisons

asyncio.run(main())
```

### Cost of Tournament

For a GRPO group of size N, the number of LLM comparisons is `N*(N-1)/2`
(doubles to `N*(N-1)` with `debiased=True`):

| Group size N | Comparisons | Debiased |
|-------------|-------------|----------|
| 4 | 6 | 12 |
| 8 | 28 | 56 |
| 16 | 120 | 240 |

Keep N ≤ 8 for cost-sensitive settings. Use a small/fast judge model.

### Position Bias Debiasing

LLMs tend to prefer the first-presented response. Set `debiased=True` to run
each pair in both orders and only count when both orderings agree. Inconsistent
pairs are skipped:

```python
strategy = GRPOTournamentEvaluationStrategy(debiased=True)

results = await strategy.execute(
    pairwise_grader.aevaluate,
    query=query,
    responses=rollouts,
)
# Each result's metadata includes valid_comparisons count
# (may be less than N-1 if some pairs were inconsistent)
```

---

## Core Concept: LISTWISE with 2 Responses

```
rank = [1, 2]  →  response_1 wins  (chosen=response_1, rejected=response_2)
rank = [2, 1]  →  response_2 wins  (chosen=response_2, rejected=response_1)
```

---

## Pattern 1 — Single Pairwise Comparison (DPO / RLAIF)

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode, GraderRank

model = OpenAIChatModel(
    model="qwen-plus",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

pairwise_grader = LLMGrader(
    model=model,
    name="preference",
    mode=GraderMode.LISTWISE,
    template="""You are an expert evaluator. Compare two responses to the same query
and decide which one is better overall.

Query: {query}

Response 1:
{response_1}

Response 2:
{response_2}

Rank the responses from best (1) to worst (2). Consider: accuracy, helpfulness,
clarity, and safety. If both are equal, prefer the more concise one.

Respond in JSON: {{"rank": [<int>, <int>], "reason": "<brief explanation>"}}""",
)

async def compare(query: str, response_a: str, response_b: str):
    result = await pairwise_grader.aevaluate(
        query=query,
        response_1=response_a,
        response_2=response_b,
    )
    # result is GraderRank: result.rank e.g. [1, 2] or [2, 1]
    if result.rank[0] == 1:
        # response_1 (response_a) ranked first → it's the winner
        return {"chosen": response_a, "rejected": response_b, "reason": result.reason}
    else:
        return {"chosen": response_b, "rejected": response_a, "reason": result.reason}

asyncio.run(compare(
    query="Explain gradient descent.",
    response_a="Gradient descent minimizes loss by following the negative gradient...",
    response_b="It's an optimization thing.",
))
```

---

## Pattern 2 — Batch Preference Labeling with GradingRunner

For building large preference datasets, use `GradingRunner` to parallelize comparisons.

### Dataset format for pairwise

Each sample must have `response_1` and `response_2` as separate fields:

```python
pairs_dataset = [
    {
        "query": "What is photosynthesis?",
        "response_1": "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "response_2": "Plants make food.",
    },
    {
        "query": "Write a Python function to reverse a list.",
        "response_1": "def reverse_list(lst): return lst[::-1]",
        "response_2": "def reverse_list(lst):\n    result = []\n    for i in range(len(lst)-1, -1, -1):\n        result.append(lst[i])\n    return result",
    },
]
```

### Run batch comparisons

```python
from openjudge.runner.grading_runner import GradingRunner

runner = GradingRunner(
    grader_configs={"preference": pairwise_grader},
    max_concurrency=16,
)

async def build_preference_dataset(pairs: list[dict]) -> list[dict]:
    results = await runner.arun(pairs)

    preference_pairs = []
    for i, result in enumerate(results["preference"]):
        pair = pairs[i]
        if isinstance(result, GraderRank):
            if result.rank[0] == 1:
                chosen, rejected = pair["response_1"], pair["response_2"]
            else:
                chosen, rejected = pair["response_2"], pair["response_1"]
            preference_pairs.append({
                "prompt": pair["query"],
                "chosen": chosen,
                "rejected": rejected,
                "reason": result.reason,
            })
        else:
            # GraderError — skip or handle
            pass

    return preference_pairs

pref_data = asyncio.run(build_preference_dataset(pairs_dataset))
```

---

## Pattern 3 — Multi-dimensional Pairwise Judge

For a more principled comparison, use a judge that evaluates on specific criteria:

```python
multi_dim_pairwise = LLMGrader(
    model=model,
    name="multi_dim_preference",
    mode=GraderMode.LISTWISE,
    template="""Compare two responses across three dimensions:
1. **Correctness**: Is the information accurate?
2. **Helpfulness**: Does it fully address the query?
3. **Safety**: Is the content appropriate?

Query: {query}

Response 1:
{response_1}

Response 2:
{response_2}

For each dimension, note which response is better. Then give a final overall ranking.

Respond in JSON:
{{
  "correctness_winner": <1 or 2>,
  "helpfulness_winner": <1 or 2>,
  "safety_winner": <1 or 2>,
  "rank": [<int>, <int>],
  "reason": "<explanation>"
}}""",
)
```

> **Note:** Even though the template asks for intermediate fields, `LLMGrader` in
> LISTWISE mode extracts only `rank` and `reason` from the JSON response.
> The extra fields are ignored but help the model reason step-by-step (chain-of-thought).

---

## Pattern 4 — Position Bias Mitigation

LLMs tend to prefer the first-presented response. Mitigate this by running each
pair twice with swapped order and only accepting when both orderings agree.

For **tournament scenarios**, use `GRPOTournamentEvaluationStrategy(debiased=True)`
(see Pattern 0 above).

For **single pairwise comparisons** (DPO preference labeling), use this helper:

```python
import asyncio
from openjudge.graders.schema import GraderRank

async def compare_debiased(
    grader: LLMGrader,
    query: str,
    response_a: str,
    response_b: str,
) -> dict | None:
    """Run A-vs-B and B-vs-A; return preference only if both agree."""
    result_ab, result_ba = await asyncio.gather(
        grader.aevaluate(query=query, response_1=response_a, response_2=response_b),
        grader.aevaluate(query=query, response_1=response_b, response_2=response_a),
    )

    if not isinstance(result_ab, GraderRank) or not isinstance(result_ba, GraderRank):
        return None

    a_wins_ab = result_ab.rank[0] == 1   # A ranked 1st when presented first
    b_wins_ba = result_ba.rank[0] == 1   # B ranked 1st when presented first

    if a_wins_ab and not b_wins_ba:
        # A wins in both orderings (consistent)
        return {"chosen": response_a, "rejected": response_b}
    elif not a_wins_ab and b_wins_ba:
        # B wins in both orderings (consistent)
        return {"chosen": response_b, "rejected": response_a}
    else:
        # Both win when first (position bias) or both lose when first — discard
        return None
```

---

## Pattern 5 — Converting Pairwise to Soft Rewards

For algorithms that accept soft preference probabilities instead of hard pairs:

```python
async def pairwise_soft_reward(
    grader: LLMGrader,
    query: str,
    response_a: str,
    response_b: str,
    n_votes: int = 5,
) -> float:
    """
    Returns P(A > B) estimated from n_votes comparisons.
    0.0 = B always wins, 1.0 = A always wins.
    """
    a_wins = 0
    tasks = [
        grader.aevaluate(query=query, response_1=response_a, response_2=response_b)
        for _ in range(n_votes)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        if isinstance(r, GraderRank) and r.rank[0] == 1:
            a_wins += 1
    return a_wins / n_votes

# reward_a - reward_b signal for PPO:
p_a_wins = await pairwise_soft_reward(grader, query, response_a, response_b, n_votes=5)
reward_a = p_a_wins          # e.g. 0.8
reward_b = 1.0 - p_a_wins   # e.g. 0.2
```

---

## Generating Preference Data at Scale

Full pipeline to generate DPO-ready preference dataset from model rollouts:

```python
import asyncio
import json
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode, GraderRank
from openjudge.runner.grading_runner import GradingRunner

async def generate_dpo_dataset(
    prompts: list[str],
    policy_responses: list[str],      # "chosen" candidates (better model)
    reference_responses: list[str],   # "rejected" candidates (weaker model)
    judge_model: OpenAIChatModel,
    concurrency: int = 16,
) -> list[dict]:
    """
    Compare policy responses vs reference responses for each prompt.
    Returns DPO-ready pairs where policy wins.
    """
    grader = LLMGrader(
        model=judge_model,
        name="preference",
        mode=GraderMode.LISTWISE,
        template="""Which response better answers the query?

Query: {query}
Response 1: {response_1}
Response 2: {response_2}

Respond in JSON: {{"rank": [<int>, <int>], "reason": "<explanation>"}}""",
    )

    dataset = [
        {"query": p, "response_1": r1, "response_2": r2}
        for p, r1, r2 in zip(prompts, policy_responses, reference_responses)
    ]

    runner = GradingRunner(
        grader_configs={"preference": grader},
        max_concurrency=concurrency,
    )
    results = await runner.arun(dataset)

    dpo_pairs = []
    for i, result in enumerate(results["preference"]):
        if not isinstance(result, GraderRank):
            continue
        if result.rank[0] == 1:
            # policy response (response_1) wins
            dpo_pairs.append({
                "prompt": prompts[i],
                "chosen": policy_responses[i],
                "rejected": reference_responses[i],
            })
        else:
            # reference wins — still a valid pair (just inverted)
            dpo_pairs.append({
                "prompt": prompts[i],
                "chosen": reference_responses[i],
                "rejected": policy_responses[i],
            })

    return dpo_pairs
```
