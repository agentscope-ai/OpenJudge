# Run Grading Tasks
> **Tip:** Grading is the process of evaluating how good responses from an AI model are. Think of it like having different teachers grade student essays - each teacher focuses on a different aspect like grammar, content, or creativity.

## What is GradingRunner?
The [GradingRunner](../../openjudge/runner/grading_runner.py) is OpenJudge's primary execution engine that orchestrates the evaluation process across multiple graders. It acts as the conductor of an orchestra, coordinating all the different graders to create a harmonious evaluation process.

GradingRunner's main purpose is to coordinate the execution of multiple graders across your dataset, providing the infrastructure needed for efficient and effective evaluations. Specifically, it provides:

- **Execution orchestration**: Managing the execution of multiple graders across your dataset
- **Data mapping services**: Transforming your data fields to match the parameter names expected by your graders
- **Concurrency management**: Controlling how many evaluations happen simultaneously to optimize resource usage
- **Result aggregation**: Combining results from multiple graders into unified scores when needed
- **Resource optimization**: Executing graders concurrently to maximize throughput

The GradingRunner encapsulates the complexity of managing concurrent evaluations, data transformations, and result aggregation. It provides a clean interface that allows you to focus on defining what you want to evaluate rather than worrying about how to execute those evaluations efficiently. With built-in support for asynchronous processing, the GradingRunner maximizes throughput when working with resource-intensive graders like large language models.

## Configuring and Using the Runner
Setting up an evaluation workflow with the GradingRunner involves configuring the graders you want to use and how they should be executed. The configuration process involves defining grader instances and mapping your dataset fields to the inputs expected by each grader. This mapping is crucial because your data format rarely matches exactly what graders expect. Additionally, you can configure aggregators to combine results from multiple graders into composite scores.

Let's begin with a simple example to understand how GradingRunner works:

```python
from openjudge.runner.grading_runner import GradingRunner
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.models.openai_chat_model import OpenAIChatModel

# Prepare your data in whatever format works for you
data = [
    {
        "query": "What is the capital of France?",
        "response": "Paris",
        "reference_answer": "Paris"
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "response": "Shakespeare",
        "reference_answer": "William Shakespeare"
    }
]

# Configure graders with mappers to connect your data fields
grader_configs = {
    "correctness": {
        "grader": CorrectnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"query": "query", "response": "response", "reference_response": "reference_answer"}
    },
    "relevance": {
        "grader": RelevanceGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"query": "query", "response": "response", "reference_response": "reference_answer"}
    }
}

# Configure aggregators to combine results
aggregators = [
    WeightedSumAggregator(name="overall_quality", weights={"correctness": 0.6, "relevance": 0.4})
]

# Run evaluation with concurrency control
runner = GradingRunner(
    grader_configs,
    aggregators=aggregators,
    max_concurrency=5
)
results = await runner.arun(data)
```

This example demonstrates several key concepts that are essential to understanding how to configure and use Runners:

### Data Mapping
The GradingRunner's mapper functionality allows you to transform your data fields to match the parameter names expected by your graders. Since your input data may not have the exact field names that your graders expect, mappers provide a way to map between your data structure and the grader's expected inputs.

In the example above:
- The CorrectnessGrader expects inputs named "query", "response", and "reference_response", so we map our data field "reference_answer" to "reference_response".
- The RelevanceGrader expects the same inputs, so we use the same mapping pattern.

Types of mappers include:
- Dictionary mappers for simple key-value mappings (e.g., `{"question": "query", "answer": "response"}`)
- Callable mappers for custom functions that transform data in more complex ways

When your field names don't align with grader expectations:

```python
# Your data structure - notice the field names differ from what graders expect
dataset = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "reference_answer": "Paris"
    }
]

# Map your fields to what graders expect
# This tells the runner how to convert your data format to what graders need
grader_configs = {
    "correctness": {
        "grader": CorrectnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {
            "query": "question",      # Grader expects "query", your data has "question"
            "response": "answer",      # Grader expects "response", your data has "answer"
            "reference_response": "reference_answer"  # Grader expects "reference_response", your data has "reference_answer"
        }
    },
    "relevance": {
        "grader": RelevanceGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {
            "query": "question",
            "response": "answer",
            "reference_response": "reference_answer"  # Grader expects "reference_response", your data has "reference_answer"
        }
    }
}
```

For more complex data structures, custom mapper functions provide flexibility:

```python
# Nested data structure - more complex than what graders expect
dataset = [
    {
        "input": {"question": "What is the capital of France?"},
        "output": {"answer": "The capital of France is Paris."}
    }
]

# Custom transformation function to flatten the data
def custom_mapper(sample):
    return {
        "query": sample["input"]["question"],
        "response": sample["output"]["answer"]
    }

grader_configs = {
    "correctness": {
        "grader": CorrectnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": custom_mapper
    }
}
```

### Aggregation Configuration
After running multiple graders, you might want to combine their results into a single score. The [aggregator submodule](../../openjudge/runner/aggregator/) provides components that take multiple grader results and combine them into a unified result:

- **WeightedSumAggregator**: Combining results using weighted averages. In our example, we assign 60% weight to helpfulness and 40% to relevance: `WeightedSumAggregator(weights={"helpfulness": 0.6, "relevance": 0.4})`
- **MaxAggregator**: Taking the maximum score among all graders
- **MinAggregator**: Taking the minimum score among all graders

These aggregators allow you to create composite scores that reflect multiple evaluation dimensions, making it easier to compare overall performance across different models or configurations.

Often you'll want to combine multiple grader results into unified scores:

```python
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

# Combine multiple perspectives into a single quality score
# Like calculating a final grade based on different subject scores
aggregator = WeightedSumAggregator(
    name="overall_score",
    weights={
        "correctness": 0.6,  # Weight correctness at 60%
        "relevance": 0.4     # Weight relevance at 40%
    }
)

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=aggregator
)
```

### Concurrency Control
The GradingRunner is designed for high-performance evaluation by managing execution concurrency:

- **Multi-Grader concurrency**: Multiple different graders execute concurrently for each data item, improving evaluation speed
- **Data concurrency**: Multiple data items are processed concurrently across all graders
- **Concurrency limits**: The `max_concurrency` parameter controls the maximum number of concurrent operations to prevent system overload. In our example, `max_concurrency=5` limits the system to processing 5 items simultaneously.

Concurrency control enables efficient processing of large datasets while maintaining system stability.

Adjust concurrency based on your resources and constraints:

```python
# For resource-intensive graders (e.g., LLM-Based)
# Lower concurrency to avoid overwhelming resources like GPU or API limits
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=5  # Process 5 samples at a time
)

# For fast, lightweight graders
# Higher concurrency for faster processing
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=32  # Process 32 samples at a time
)
```

Performance optimization is especially important when working with large language model-based graders, which can be slow and resource-intensive. The right balance of concurrency can dramatically reduce evaluation time without overwhelming your system resources.

Finding the optimal concurrency level depends on your hardware resources, the types of graders you're using, and any external rate limits (like API quotas). Experimentation is often needed to find the sweet spot for your specific use case.

## Next Steps

- [Grader Analysis](grader_analysis.md) — Analyze results and generate validation reports
- [Validate Graders](../validating_graders/overview.md) — Assess the quality of your evaluations
- [Refine Data Quality](../applications/data_refinement.md) — Improve model outputs using grader feedback

