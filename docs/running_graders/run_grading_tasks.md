# :rocket: Run Grading Tasks

The [GradingRunner](../../rm_gallery/core/runner/grading_runner.py) is RM-Gallery's core execution engine, orchestrating the entire evaluation process. Whether you're evaluating a few samples or processing thousands, understanding how to effectively use the GradingRunner is key to getting the most out of RM-Gallery.

## Getting Started with GradingRunner

At its core, the GradingRunner takes your dataset and runs it through configured graders. But there's more to it than that. Let's start with a simple example and build up to more complex scenarios.

### Basic Evaluation Workflow

Here's how to set up a basic evaluation with multiple graders:

```python
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.graders.common.accuracy import AccuracyGrader

# Your evaluation data
dataset = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "query": "What is 2+2?",
        "response": "2+2 equals 4."
    }
]

# Configure the graders you want to run
grader_configs = {
    "helpfulness": HelpfulnessGrader(),
    "accuracy": AccuracyGrader()
}

# Create and run the evaluation
runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Process the results
for grader_name, grader_results in results.items():
    print(f"\nResults from {grader_name}:")
    for i, result in enumerate(grader_results):
        if hasattr(result, 'score'):
            print(f"  Sample {i+1}: Score = {result.score}")
        # Handle other result types...
```

This basic workflow works well for straightforward evaluations, but real-world scenarios often require more sophisticated handling.

## :twisted_rightwards_arrows: Bridging Data and Graders

One common challenge is that your data rarely matches exactly what graders expect. That's where data mappers come in handy.

### Field Mapping

When your field names don't align with grader expectations:

```python
# Your data structure
dataset = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "reference_answer": "Paris"
    }
]

# Map your fields to what graders expect
grader_configs = {
    "helpfulness": {
        "grader": HelpfulnessGrader(),
        "mapper": {
            "query": "question",
            "response": "answer"
        }
    },
    "relevance": {
        "grader": RelevanceGrader(),
        "mapper": {
            "query": "question",
            "response": "answer",
            "reference": "reference_answer"
        }
    }
}
```

### Complex Transformations

For more complex data structures, custom mapper functions provide flexibility:

```python
# Nested data structure
dataset = [
    {
        "input": {"question": "What is the capital of France?"},
        "output": {"answer": "The capital of France is Paris."}
    }
]

# Custom transformation function
def custom_mapper(sample):
    return {
        "query": sample["input"]["question"],
        "response": sample["output"]["answer"]
    }

grader_configs = {
    "helpfulness": {
        "grader": HelpfulnessGrader(),
        "mapper": custom_mapper
    }
}
```

## :rocket: Optimizing Performance

When dealing with large datasets or resource-intensive graders, performance becomes critical.

### Controlling Concurrency

Adjust concurrency based on your resources and constraints:

```python
# For resource-intensive graders (e.g., LLM-based)
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=5  # Lower to avoid overwhelming resources
)

# For fast, lightweight graders
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=32  # Higher for faster processing
)
```

### Combining Results with Aggregators

Often you'll want to combine multiple grader results into unified scores:

```python
from rm_gallery.core.runner.aggregator.weighted_sum import WeightedSumAggregator

# Combine multiple perspectives into a single quality score
aggregator = WeightedSumAggregator(
    weights={
        "helpfulness": 0.6,
        "relevance": 0.4
    }
)

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=aggregator
)
```

## :ambulance: Handling Real-World Challenges

Production environments bring unique challenges that require careful consideration.

### Error Resilience

Graders can fail for various reasons, but your pipeline shouldn't break:

```python
results = await runner.arun(dataset)

# Handle errors gracefully
for grader_name, grader_results in results.items():
    for i, result in enumerate(grader_results):
        if hasattr(result, 'error'):
            print(f"Error in {grader_name} for sample {i+1}: {result.error}")
            # Log, retry, or skip as appropriate
        else:
            # Process successful results
            process_result(result)
```

### Different Evaluation Modes

Graders can work in different modes depending on your needs:

```python
from rm_gallery.core.graders.schema import GraderMode

# Pointwise: Evaluate each sample independently
pointwise_grader = LLMGrader(mode=GraderMode.POINTWISE, ...)

# Listwise: Rank multiple samples together
listwise_grader = LLMGrader(mode=GraderMode.LISTWISE, ...)
```

## :zap: Best Practices for Large-Scale Evaluations

When running large evaluations, keep these tips in mind:

1. **Batch Processing**: Process data in chunks to manage memory usage
2. **Progress Monitoring**: Track progress for long-running evaluations
3. **Resource Management**: Monitor CPU, memory, and API usage
4. **Checkpointing**: Save intermediate results to recover from failures
5. **Rate Limiting**: Respect API rate limits for external services

## :tada: Putting It All Together

A robust evaluation setup combines all these elements:

```python
# Comprehensive configuration
runner = GradingRunner(
    grader_configs={
        "helpfulness": {
            "grader": HelpfulnessGrader(),
            "mapper": {"query": "question", "response": "answer"}
        },
        "accuracy": {
            "grader": AccuracyGrader(),
            "mapper": {"question": "question", "response": "answer"}
        },
        "relevance": RelevanceGrader()
    },
    max_concurrency=10,
    aggregators=WeightedSumAggregator(weights={"helpfulness": 0.5, "accuracy": 0.3, "relevance": 0.2})
)

# Run with error handling
try:
    results = await runner.arun(large_dataset)
    # Process and analyze results
except Exception as e:
    print(f"Evaluation failed: {e}")
```

Once you've mastered running grading tasks, you'll want to [generate validation reports](../validating_graders/generate_validation_reports.md) to assess the quality of your evaluations or [refine data quality](../applications/refine_data_quality.md) using your evaluation insights.