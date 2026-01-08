# Langfuse Integration

This tutorial guides you through building powerful external evaluation pipelines for Langfuse using OpenJudge. By integrating these two tools, you can leverage OpenJudge's 50+ built-in graders to perform comprehensive automated quality evaluation of your LLM applications.

## Overview

While Langfuse provides built-in evaluation features, external evaluation pipelines offer additional advantages:

- **Flexible Triggering**: Trigger evaluations at any time, independent of application runtime
- **Rich Evaluation Dimensions**: OpenJudge provides 50+ graders covering quality, safety, format, agent behaviors, and more
- **Extensibility**: Easily add custom evaluation logic to meet specific business requirements
- **Batch Processing**: Efficiently process large volumes of historical traces with support for scheduled tasks and incremental evaluation

The integration follows a simple three-step flow:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│    Langfuse     │      │    OpenJudge    │      │    Langfuse     │
│    (Traces)     │─────▶│  (Evaluation)   │─────▶│    (Scores)     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

1. **Fetch Traces** — Pull traces from Langfuse using `api.trace.list()`
2. **Run Evaluation** — Score traces with OpenJudge graders via `graders.aevaluate()`
3. **Send Results** — Push scores back to Langfuse via `create_score()`

## Prerequisites

!!! warning "Required Configuration"
    Make sure to set all required environment variables before running the code.

**Install dependencies:**

```bash
pip install py-openjudge langfuse
```

**Configure environment variables:**

```bash
# Langfuse authentication
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Or your self-hosted URL

# OpenAI API configuration (required for LLM-based graders)
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
```

**Initialize the Langfuse client:**

```python
import os
from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Verify connection
assert langfuse.auth_check(), "Langfuse authentication failed"
```

## Integration Steps

### Step 1: Prepare Traces

If you already have traces in Langfuse, you can skip this step and proceed directly to Step 2. The following code creates synthetic traces for testing purposes:

```python
from langfuse import Langfuse, observe

# Initialize Langfuse client
langfuse = Langfuse()

@observe(name="simple_qa_app")
def simple_qa_app(question: str) -> str:
    """Simulate a simple QA application"""
    # Simulated LLM responses
    answers = {
        "What is machine learning?": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed.",
        "What is the capital of France?": "The capital of France is Paris.",
        "Explain quantum computing": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, enabling parallel computation.",
    }
    return answers.get(question, "I don't know the answer to that question.")

# Create test traces
test_questions = [
    "What is machine learning?",
    "What is the capital of France?",
    "Explain quantum computing",
]

for question in test_questions:
    response = simple_qa_app(question)
    print(f"Q: {question}\nA: {response}\n")

# Ensure all traces are sent
langfuse.flush()
```

### Step 2: Fetch Traces

Now fetch the traces from Langfuse that you want to evaluate:

```python
def fetch_traces_for_evaluation(
    limit: int = 100,
    tags: list[str] | None = None,
) -> list[dict]:
    """
    Fetch traces from Langfuse for evaluation.

    Args:
        limit: Maximum number of traces to fetch
        tags: Optional tag filter

    Returns:
        List of trace dictionaries
    """
    # Fetch traces using API
    response = langfuse.api.trace.list(
        limit=limit,
        tags=tags,
    )

    result = []
    for trace in response.data:
        # Only include traces with input and output
        if trace.input and trace.output:
            trace_dict = {
                "id": trace.id,
                "input": trace.input,
                "output": trace.output,
                "metadata": trace.metadata or {},
            }

            # Add expected output if available in metadata
            if trace.metadata and "expected" in trace.metadata:
                trace_dict["expected"] = trace.metadata["expected"]

            result.append(trace_dict)

    return result
```

??? tip "Advanced: Fetch by Time Range"
    In production, you often want to filter traces by time range:

    ```python
    from datetime import datetime, timedelta

    def fetch_recent_traces(
        hours_back: int = 24,
        limit: int = 100,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch traces from the last N hours.

        Args:
            hours_back: Number of hours to look back
            limit: Maximum number of traces to fetch
            tags: Optional tag filter

        Returns:
            List of trace dictionaries
        """
        from_timestamp = datetime.now() - timedelta(hours=hours_back)

        response = langfuse.api.trace.list(
            limit=limit,
            tags=tags,
            from_timestamp=from_timestamp,
        )

        result = []
        for trace in response.data:
            if trace.input and trace.output:
                result.append({
                    "id": trace.id,
                    "input": trace.input,
                    "output": trace.output,
                    "metadata": trace.metadata or {},
                })

        return result
    ```

### Step 3: Run Evaluation

With the traces fetched, you can now evaluate them using OpenJudge graders. First, select the appropriate grader based on your evaluation needs:

| Evaluation Scenario | Recommended Grader | Type | Description |
|---------------------|-------------------|------|-------------|
| Response relevance | `RelevanceGrader` | LLM-Based | Evaluates response-query relevance (1-5) |
| Content safety | `HarmfulnessGrader` | LLM-Based | Detects harmful content (1-5) |
| Hallucination detection | `HallucinationGrader` | LLM-Based | Identifies fabricated information (1-5) |
| Instruction following | `InstructionFollowingGrader` | LLM-Based | Checks instruction compliance (1-5) |
| Answer correctness | `CorrectnessGrader` | LLM-Based | Compares with reference answer (1-5) |
| Text similarity | `SimilarityGrader` | Code-Based | Computes text similarity (0-1) |
| JSON validation | `JsonValidatorGrader` | Code-Based | Validates JSON syntax (0/1) |
| Agent tool calls | `ToolCallAccuracyGrader` | LLM-Based | Evaluates tool call quality (1-5) |

For the complete list of 50+ built-in graders, see [Built-in Graders Overview](../built_in_graders/overview.md).

=== "Single Grader (Quick Start)"

    The simplest approach is to evaluate traces one by one with a single grader:

    ```python
    import asyncio
    from openjudge.models import OpenAIChatModel
    from openjudge.graders.common.relevance import RelevanceGrader
    from openjudge.graders.schema import GraderScore, GraderError

    async def evaluate_single_trace():
        """Evaluate traces using a single grader"""

        # Initialize model and grader
        model = OpenAIChatModel(model="qwen3-32b")
        grader = RelevanceGrader(model=model)

        # Fetch traces
        traces = fetch_traces_for_evaluation(limit=10)

        for trace in traces:
            try:
                # Run evaluation
                # Note: RelevanceGrader uses 'query' and 'response' parameters
                result = await grader.aevaluate(
                    query=trace["input"],
                    response=trace["output"],
                )

                # Process result and send to Langfuse
                if isinstance(result, GraderScore):
                    langfuse.create_score(
                        trace_id=trace["id"],
                        name="relevance",
                        value=result.score,
                        comment=result.reason,
                    )
                    print(f"✓ Trace {trace['id'][:8]}... scored: {result.score}")
                elif isinstance(result, GraderError):
                    print(f"✗ Trace {trace['id'][:8]}... error: {result.error}")

            except Exception as e:
                print(f"✗ Error evaluating trace {trace['id']}: {e}")

        # Ensure all scores are sent
        langfuse.flush()

    # Run evaluation
    asyncio.run(evaluate_single_trace())
    ```

=== "Batch Evaluation (Recommended)"

    For large numbers of traces, use `GradingRunner` for efficient concurrent batch evaluation. This approach supports multiple graders, field mapping, and score aggregation:

    ```python
    import asyncio
    from openjudge.models import OpenAIChatModel
    from openjudge.graders.common.relevance import RelevanceGrader
    from openjudge.graders.common.harmfulness import HarmfulnessGrader
    from openjudge.graders.text.similarity import SimilarityGrader
    from openjudge.graders.schema import GraderScore, GraderRank, GraderError
    from openjudge.runner.grading_runner import GradingRunner
    from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

    async def batch_evaluate_traces():
        """Batch evaluate traces using GradingRunner"""

        # Initialize model
        model = OpenAIChatModel(model="qwen3-32b")

        # Configure multiple graders with field mappers
        # Map trace fields to grader expected parameters
        runner = GradingRunner(
            grader_configs={
                "relevance": (
                    RelevanceGrader(model=model),
                    {"query":"input", "response":"output"}  # Map input->query, output->response
                ),
                "harmfulness": (
                    HarmfulnessGrader(model=model),
                    {"query":"input", "response":"output"}
                )
            },
            max_concurrency=10,  # Control concurrency to avoid rate limits
            show_progress=True,  # Show progress bar
            # Optional: Add aggregators to combine scores into a composite score
            aggregators=[
                WeightedSumAggregator(
                    name="overall_quality",
                    weights={
                        "relevance": 0.7,    # 70% weight on relevance
                        "harmfulness": 0.3,  # 30% weight on safety
                    }
                )
            ],
        )

        # Fetch traces
        traces = fetch_traces_for_evaluation(limit=50)

        if not traces:
            print("No traces to evaluate")
            return

        # Prepare evaluation data
        evaluation_data = []
        trace_id_mapping = {}  # Map index to trace_id

        for i, trace in enumerate(traces):
            eval_item = {
                "input": trace["input"],
                "output": trace["output"],
            }
            # Add expected output as reference if available
            if trace.get("expected"):
                eval_item["expected"] = trace["expected"]

            evaluation_data.append(eval_item)
            trace_id_mapping[i] = trace["id"]

        # Run batch evaluation
        try:
            results = await runner.arun(evaluation_data)

            # Send results back to Langfuse
            # results contains individual grader scores + aggregated "overall_quality" score
            scores_sent = 0
            for grader_name, grader_results in results.items():
                for i, result in enumerate(grader_results):
                    trace_id = trace_id_mapping[i]
                    print(f"Sending {grader_name} score for trace {trace_id}")
                    send_result_to_langfuse(trace_id, grader_name, result)
                    scores_sent += 1


            print(f"✓ Successfully sent {scores_sent} scores for {len(traces)} traces")

        except Exception as e:
            print(f"✗ Batch evaluation failed: {e}")

        # Ensure all scores are sent
        langfuse.flush()
    ```

    The helper function to send results back to Langfuse:

    ```python
    def send_result_to_langfuse(trace_id: str, grader_name: str, result) -> None:
        """Send evaluation result to Langfuse"""

        if isinstance(result, GraderScore):
            langfuse.create_score(
                trace_id=trace_id,
                name=grader_name,
                value=result.score,
                comment=result.reason[:500] if result.reason else "",
            )
        elif isinstance(result, GraderRank):
            # For ranking results, store the first rank position
            langfuse.create_score(
                trace_id=trace_id,
                name=grader_name,
                value=float(result.rank[0]) if result.rank else 0.0,
                comment=result.reason[:500] if result.reason else "",
            )
        elif isinstance(result, GraderError):
            # For errors, record as 0 score with error description
            langfuse.create_score(
                trace_id=trace_id,
                name=f"{grader_name}_error",
                value=0.0,
                comment=f"Evaluation error: {result.error}"[:500],
            )

    # Run batch evaluation
    asyncio.run(batch_evaluate_traces())
    ```

### Step 4: View Results

After running the evaluation, you can view the scores in your Langfuse dashboard:

![Langfuse Score Results](../images/langfuse_score_result.png)

The scores include individual grader results (e.g., `relevance`, `harmfulness`) and aggregated composite scores (e.g., `overall_quality`).

## Related Resources

- [OpenJudge Built-in Graders](../built_in_graders/overview.md) — Explore 50+ available graders
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific evaluation logic
- [Langfuse Tracing Documentation](https://langfuse.com/docs/tracing) — Learn about Langfuse tracing
- [Langfuse Scores API](https://langfuse.com/docs/scores) — Scores API reference
