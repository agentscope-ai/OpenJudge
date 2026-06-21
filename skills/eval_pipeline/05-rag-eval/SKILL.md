---
name: rag-eval
description: >
  Use when the user has a RAG (Retrieval-Augmented Generation) system and wants to
  evaluate its quality — separating retrieval issues from generation issues. Also use
  when the user mentions RAG evaluation, faithfulness checking, hallucination detection
  in RAG, retrieval quality, chunking optimization, or "is my RAG pipeline working."
  Outputs a diagnostic matrix that pinpoints whether problems are in retrieval or generation.
---

# RAG Eval

Evaluate RAG systems by diagnosing retrieval and generation separately. A single
"RAG accuracy" number hides whether the problem is finding the right documents or
using them correctly. This skill separates them so you know what to fix.

## When to Activate

- User has a RAG pipeline (retriever + generator) with traces
- User wants to know if their RAG system hallucinates
- User is optimizing chunking strategy and needs before/after comparison
- User wants to build a RAG evaluation dataset

## Checklist

You MUST create a task for each item and complete them in order:

1. **Load RAG traces** — validate query + context + answer triples
2. **Separate retrieval vs generation** — determine which layers to evaluate
3. **Run faithfulness evaluation** — is the answer grounded in retrieved docs?
4. **Run retrieval evaluation** — are the right documents retrieved?
5. **Build diagnostic matrix** — cross-tabulate to find root cause
6. **Output findings** — prioritized issues with concrete fixes

## Fast path: run the bundled script

Once each trace has a faithfulness judgment (and ideally a retrieval signal), build the
retrieval-vs-generation diagnostic matrix with the bundled, tested script
(`scripts/rag_diagnostic.py`, standard library only, **no OpenJudge dependency**):

```bash
python scripts/rag_diagnostic.py --traces traces.jsonl
```

Trace rows: `{"faithful":bool}` or `{"faithfulness_score":1-5}` (>= 4 = faithful), plus an
optional retrieval signal `{"retrieval_good":bool}` or `{"recall_at_k":0-1}` (>= 0.5 = good).
It prints the generation faithful/hallucinating split, the 2×2 matrix when a retrieval signal
is present, and the primary issue (retrieval vs generation). `--self-test` to verify it.

Steps below explain how to separate the layers and produce the faithfulness/retrieval signals
(with OpenJudge graders or any judge).

## Step 1: Load RAG Traces

The minimum data needed per trace:

```python
# Each trace must contain:
trace = {
    "query": "What is the return policy?",
    # retrieved_docs are dicts with id + text (the id is required for retrieval
    # metrics like Recall@k; the text is required for the faithfulness check).
    "retrieved_docs": [
        {"id": "doc_1", "text": "Returns are accepted within 30 days..."},
        {"id": "doc_2", "text": "Refunds are issued to the original..."},
    ],
    "answer": "You can return items within 30 days for a full refund.",
    "reference_answer": "Our 30-day return policy allows full refunds.",  # optional
    "gold_doc_ids": ["doc_1", "doc_3"],  # optional — which docs should have been retrieved
}
```

If your traces only have raw strings (no ids), wrap them first so retrieval metrics work:
`retrieved_docs = [{"id": f"d{i}", "text": t} for i, t in enumerate(raw_strings)]`.

Validate data completeness and report any gaps. If > 20% of traces are missing key fields,
ask the user to confirm the schema before proceeding.

## Step 2: Separate Retrieval vs Generation

RAG failures come from two independent sources:

| Source | What goes wrong | Metric to use |
|--------|----------------|---------------|
| **Retrieval** | Wrong/missing documents returned | Recall@k, Precision@k, MRR |
| **Generation** | Model misuses or fabricates beyond docs | Faithfulness (HallucinationGrader) |
| **Generation** | Answer doesn't address the query | Relevance (RelevanceGrader) |

Why separate them? A system with perfect retrieval but poor generation needs prompt
engineering. A system with poor retrieval needs chunking/embedding work. Treating
them as one problem wastes effort.

## Step 3: Faithfulness Evaluation

Use OpenJudge `HallucinationGrader` to check if the answer stays grounded in
retrieved documents:

```python
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.runner.grading_runner import GradingRunner

faithfulness_grader = HallucinationGrader(model=model)

runner = GradingRunner(
    grader_configs={"faithfulness": faithfulness_grader},
    max_concurrency=8,
)

# Dataset format for HallucinationGrader
dataset = [
    {
        "query": trace["query"],
        "response": trace["answer"],
        "context": "\n\n".join(doc["text"] for doc in trace["retrieved_docs"]),
    }
    for trace in traces
]

results = await runner.arun(dataset)

# HallucinationGrader scores 1-5 (5 = no hallucination, fully grounded)
# Binarize: score >= 4 → faithful, score < 4 → hallucination
```

Also evaluate **answer relevance** — does the response actually address the query?

```python
from openjudge.graders.common.relevance import RelevanceGrader

relevance_grader = RelevanceGrader(model=model)
```

## Step 4: Retrieval Evaluation

If gold_doc_ids are available, compute retrieval metrics:

```python
def recall_at_k(retrieved_docs, gold_ids, k=5):
    """Fraction of gold docs found in top-k retrieved docs."""
    retrieved_ids = set(doc["id"] for doc in retrieved_docs[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return None
    return len(retrieved_ids & gold_set) / len(gold_set)

def precision_at_k(retrieved_docs, gold_ids, k=5):
    """Fraction of top-k docs that are relevant."""
    retrieved_ids = set(doc["id"] for doc in retrieved_docs[:k])
    gold_set = set(gold_ids)
    if not retrieved_ids:
        return 0
    return len(retrieved_ids & gold_set) / len(retrieved_ids)

def mrr(retrieved_docs, gold_ids):
    """Mean Reciprocal Rank — how early the first relevant doc appears."""
    for i, doc in enumerate(retrieved_docs):
        if doc["id"] in gold_ids:
            return 1 / (i + 1)
    return 0
```

If gold_doc_ids are NOT available, use a semantic relevance judge as a proxy:
check whether each retrieved doc is semantically related to the query.

### Chunking Optimization

If retrieval is the bottleneck, try a grid search over chunk size and overlap:

```python
chunk_configs = [
    {"chunk_size": 256, "overlap": 32},
    {"chunk_size": 512, "overlap": 64},
    {"chunk_size": 1024, "overlap": 128},
    {"chunk_size": 512, "overlap": 128},
]

for config in chunk_configs:
    # Re-chunk, re-embed, re-retrieve, compute Recall@5
    ...
```

## Step 5: Diagnostic Matrix

The most valuable output — cross-tabulate retrieval vs generation results:

```
                    Generation: Faithful    Generation: Hallucinating
Retrieval: Good          ████████ 62%            ██ 10%
                         (system works)          (generator problem)

Retrieval: Poor          █ 8%                   ██████ 20%
                         (lucky guess)           (both broken)
```

Interpretation:
- **62% (Good/Faithful)**: System working correctly
- **10% (Good/Hallucinating)**: Generator is misusing or fabricating despite having
  the right docs. Fix: prompt engineering, few-shot examples, or model upgrade.
- **8% (Poor/Faithful)**: Generator got the right answer from wrong docs — likely
  using parametric knowledge, not retrieval. Unreliable.
- **20% (Poor/Hallucinating)**: Both layers broken. Fix retrieval first, then generation.

## Step 6: Output

Present findings with concrete recommendations:

```
RAG Evaluation Results (500 traces):

Layer Results:
  Faithfulness:     76% pass  (95% CI: [72%, 80%])
  Answer Relevance: 82% pass  (95% CI: [78%, 85%])
  Retrieval Recall@5: 70%    (95% CI: [66%, 74%])

Diagnostic Matrix:
                    Faithful    Hallucinating
  Retrieval Good      62%          10%  ← 10% of answers hallucinate despite good docs
  Retrieval Poor       8%          20%  ← 20% both broken

Primary issue: 10% hallucination rate even with good retrieval.
  → Generator is ignoring or misreading retrieved documents.
  → Recommendation: add "cite specific document passages" to generation prompt.
  → Re-evaluate faithfulness after prompt change.

Secondary issue: 20% of traces have both poor retrieval and hallucination.
  → Recommendation: optimize chunking first (try smaller chunks with more overlap).
  → Re-run retrieval eval after chunking changes.
```

## Common Mistakes

- **Using a single "RAG accuracy" metric.** This is the cardinal sin of RAG eval.
  You can't fix what you can't localize. Always separate retrieval from generation.
- **Skipping faithfulness because correctness looks OK.** An answer can be correct
  (matches reference) but unfaithful (not derived from retrieved docs). This means
  the generator is using parametric knowledge — which will fail silently on a different
  knowledge domain.
- **Optimizing generation before retrieval.** If retrieval is broken, no amount of
  prompt engineering will fix the answers. Fix the data pipeline first.
- **No adversarial questions in the eval set.** RAG systems look great on factual
  lookups but fail on queries that require synthesizing across multiple chunks.
  Include multi-hop and comparative questions.

## Next Skills

After `05-rag-eval`:
- **`03-align-human`**: Calibrate the faithfulness judge against human judgments.
- **`01-eval-design`**: Build a more comprehensive RAG eval dataset with adversarial queries.
- **`06-prompt-regression`**: If you change the generation prompt, measure before/after.