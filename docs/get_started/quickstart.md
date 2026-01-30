# Quick Start

Get started with OpenJudge in 5 minutes. This guide walks you through installation, environment setup, and running your first evaluation.

## Installation

=== "From PyPI"

    ```bash
    # Install with Standard dependencies from PyPI
    pip install py-openjudge

    # For development purposes, install with dev dependencies
    pip install -e .[dev]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/agentscope-ai/OpenJudge.git
    cd OpenJudge

    # Install based on your needs:
    pip install -e .        # Standard installation
    pip install -e .[dev]   # With development dependencies
    pip install -e .[verl]  # With VerL option for training scenarios
    ```

=== "Docker"

    ```bash
    git clone https://github.com/agentscope-ai/OpenJudge.git
    cd OpenJudge

    # Build the Docker image
    docker build -f docker/Dockerfile.base -t openjudge:latest .

    # Run the container
    docker run -it \
        -v $(pwd):/workspace/OpenJudge \
        --name openjudge \
        openjudge:latest
    ```

> **Tips:**
> OpenJudge requires Python version >=3.10 and <3.13. For best compatibility, we recommend using Python 3.10 or 3.11.


## Configure Environment

For LLM-Based graders, you need to configure API credentials. OpenJudge uses the OpenAI-compatible API format.

=== "Environment Variables (Recommended)"

    ```bash
    # Set environment variables in your terminal

    # OpenAI
    export OPENAI_API_KEY="sk-your-api-key"
    export OPENAI_BASE_URL="https://api.openai.com/v1"

    # DashScope (Qwen)
    export OPENAI_API_KEY="sk-your-dashscope-key"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ```

=== "Pass Directly in Code"

    ```python
    # Pass credentials directly when creating the model
    from openjudge.models import OpenAIChatModel

    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="sk-your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    ```

!!! note "Security Best Practice"
    Environment variables are more secure and convenient. The model will automatically use `OPENAI_API_KEY` and `OPENAI_BASE_URL` if set.


## Choose a Grader for Your Scenario

Suppose you're building a QA system and want to evaluate: **Does the AI assistant's response actually answer the user's question?**

This is a **relevance** evaluation task. OpenJudge provides the `RelevanceGrader` for exactly this purpose—it scores how well a response addresses the query on a 1-5 scale.

| Your Scenario | Recommended Grader |
|---------------|-------------------|
| Does the response answer the question? | `RelevanceGrader` |
| Is the response harmful or unsafe? | `HarmfulnessGrader` |
| Does the response follow instructions? | `InstructionFollowingGrader` |
| Is the response factually correct? | `CorrectnessGrader` |
| Does the response contain hallucinations? | `HallucinationGrader` |

For a complete list of available graders, see [Built-in Graders](../built_in_graders/overview.md).

In this quickstart, we'll use `RelevanceGrader` to evaluate a QA response.

## Choose an Evaluation Strategy for Your Grader

An evaluation strategy enhances the reliability and robustness of your grader by applying different techniques to improve the quality of the evaluation. Different types of graders may benefit from different evaluation strategies depending on your scenario requirements.
OpenJudge provides several evaluation strategies:

| Strategy | Purpose | Best For |
|----------|---------|----------|
| `DirectEvaluationStrategy` | Default strategy, single evaluation | Fastest option, good for initial testing |
| `AverageEvaluationStrategy` | Numerical averaging over multiple evaluations | Numeric scores, reducing noise |
| `VotingEvaluationStrategy` | Majority voting over multiple evaluations | Categorical outputs, increasing reliability |

In this quickstart, we'll use `AverageEvaluationStrategy` for `RelevanceGrader`.

!!! note "What is an Evaluation Strategy?"
    An **Evaluation Strategy** defines how the grader process is executed to enhance reliability and robustness. Learn more in [Core Concepts](core_concepts.md).

## Prepare Your Data

Prepare a dictionary with `query` and `response` fields. These field names correspond to the input parameters of the grader's `aevaluate()` method:

```python
data = {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. It uses algorithms to build models that can make predictions or decisions.",
}
```


## Initialize Grader

Create the `RelevanceGrader` to evaluate how well the response addresses the query:

```python
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.evaluation_strategy.average_evaluation_strategy import AverageEvaluationStrategy

# Create the judge model (uses OPENAI_API_KEY and OPENAI_BASE_URL from env)
model = OpenAIChatModel(model="qwen3-32b")

# Create the evaluation strategy optionally
strategy = AverageEvaluationStrategy(num_evaluations=5)

# Create the grader
grader = RelevanceGrader(model=model, strategy=strategy)
```

!!! note "What is a Grader?"
    A **Grader** is the core evaluation component in OpenJudge. It takes a query-response pair and returns a score with an explanation. Learn more in [Core Concepts](core_concepts.md).


## Run Evaluation

All graders use async/await. Evaluate your data with `aevaluate()`:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(model="qwen3-32b")
    grader = RelevanceGrader(model=model)

    # Prepare data
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. It uses algorithms to build models that can make predictions or decisions.",
    }

    # Run evaluation
    result = await grader.aevaluate(**data)

    # Print result
    print(result)

asyncio.run(main())
```

**Output:**

```
GraderScore(
    name='relevance',
    score=5.0,
    reason="The response directly and clearly defines machine learning as a subset of artificial intelligence, explains its purpose (learning patterns from data without explicit programming), and mentions the use of algorithms to build predictive models. It is concise, on-topic, and fully addresses the user's question."
)
```


## Understanding the Output

The `RelevanceGrader` returns a `GraderScore` object with the following fields:

| Field | Description | Example Value |
|-------|-------------|---------------|
| `name` | Identifier of the grader | `"relevance"` |
| `score` | Relevance score from 1 (irrelevant) to 5 (perfectly relevant) | `5.0` |
| `reason` | LLM-generated explanation for the score | `"The response directly and clearly..."` |

**Score Interpretation:**

- **5 (Perfectly relevant)**: Response completely fulfills the query, accurately answering the question
- **4 (Highly relevant)**: Response largely meets requirements, possibly missing minor details
- **3 (Partially relevant)**: Response has some connection but doesn't fully meet requirements
- **2 (Weakly relevant)**: Response has only weak connection, low practical value
- **1 (Irrelevant)**: Response is completely unrelated or contains misleading information

In this example, the response received a score of **5** because it directly defines machine learning, explains the core mechanism, and provides relevant context—fully satisfying the user's query.


## Next Steps

- [Core Concepts](core_concepts.md) — Understand graders, scoring modes, and result types
- [Built-in Graders](../built_in_graders/overview.md) — Explore all available graders
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build your own evaluation logic
