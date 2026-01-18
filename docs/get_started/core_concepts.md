# Core Concepts

OpenJudge is an advanced evaluation system designed for AI model assessment. This guide explains the fundamental components of the system: Graders, Runners, and Analyzers, and how they work together to provide comprehensive model evaluation. You'll learn what Graders are, why they matter, how Graders work to evaluate AI models, how Runners use Graders to orchestrate complex evaluation tasks, and how to analyze Grader results to assess evaluation quality.

## What Is a Grader

In the era of advanced AI systems, especially large language models (LLMs), having robust evaluation and reward mechanisms is critical for both measuring performance and guiding improvements.

**Evaluation** refers to the process of assessing model outputs against specific criteria, quantifying performance across dimensions such as accuracy, helpfulness, safety, and coherence. Evaluation provides objective measurements that help developers understand how well their models are performing on specific tasks.

**Reward** mechanisms, on the other hand, provide signals that guide model training through techniques like Reinforcement Learning from Human Feedback (RLHF). These reward signals enable automated optimization, allowing systems to self-improve without constant human intervention by providing feedback on the quality of model outputs.

!!! info "Terminology: Judge Model vs Reward Model"
    In OpenJudge, we use **judge model** to refer to models trained for evaluation. This is the same concept as **reward model** commonly used in RLHF literature. Both terms describe models that assess and score AI outputs—we prefer "judge model" to emphasize the evaluation and assessment role.

The OpenJudge framework unifies these two critical functions under a single abstraction: the Grader. A Grader is a modular, standardized component that can function as either an evaluator or a reward generator depending on your use case. As an **evaluator**, a Grader assesses model outputs against specific criteria. As a **reward generator**, a Grader provides signals that guide model training. This unified approach provides a consistent interface that simplifies the process of building, managing, and deploying both evaluation and reward systems, transforming raw model outputs into meaningful, quantifiable assessments that serve as the foundation for systematic model evaluation and automated model improvement.

## Why Graders Matter

Graders play a critical role in AI development by providing objective, consistent measurement tools that drive improvement. Without reliable assessment tools powered by Graders, distinguishing between successful and unsuccessful experiments becomes nearly impossible, making it difficult to achieve consistent progress.

Graders solve key challenges in evaluation by:

- **Providing unified evaluation framework**: A single interface that handles both evaluation and reward generation, supporting diverse assessment methods while enabling seamless integration of different evaluation strategies.
- **Enabling modular architecture**: Allowing individual evaluation components to be developed, modified, or replaced without disrupting the overall system, facilitating targeted assessment of specific capabilities.
- **Standardizing assessment approaches**: Supporting both quantitative metrics (accuracy, precision, BLEU scores) and qualitative evaluations (helpfulness, safety, appropriateness) through consistent interfaces.

This makes Graders particularly valuable for:
- **Component evaluation and selection**: Comparing different AI components like language models, retrieval systems, or agent configurations using your specific data to identify optimal choices.
- **Performance diagnosis**: Locating specific areas where applications underperform, facilitating targeted debugging and refinement of both models and agents.
- **Ongoing performance monitoring**: Tracking application behavior over time to detect issues such as data drift, model degradation, or evolving user needs.

## How Graders Work

Graders are the fundamental building blocks in OpenJudge. They're standardized components that take input data (like questions and model responses) and produce evaluation results. You can think of them as black boxes that transform model outputs into meaningful scores or rankings.

The evaluation process with a Grader involves three key stages:

### Input Data Specification

Graders accept input as a JSON dictionary containing all relevant fields needed for evaluation. Common input fields include:

| Field | Description |
|-------|-------------|
| `query` | The input prompt or question |
| `response` | The model's response to evaluate |
| `reference_response` | Ground truth or reference answer |
| `context` | Additional context for the evaluation |

Flexible [data mapping](#data-mapping) ensures your existing data formats can be easily adapted to work with graders.

### Assessment Approaches

Based on your evaluation purpose, you can choose the appropriate assessment approach:

- **Code-Based graders**: For objective, quantitative evaluations using the [FunctionGrader](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/graders/function_grader.py), these graders use predefined functions or algorithms to compute scores. These graders are deterministic and fast, ideal for metrics like exact match, format validation, or simple checks. This approach is best suited for quantitative analysis where you need consistent, reproducible results based on mathematical or logical operations.

```python
# A simple function grader that checks if response contains reference answer
def contains_reference(response, reference):
    return float(reference.lower() in response.lower())

contains_grader = FunctionGrader(contains_reference)
```

- **LLM-Based graders**: For subjective, qualitative evaluations using the [LLMGrader](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/graders/llm_grader.py), these graders leverage large language models to perform sophisticated evaluations. These graders can assess complex qualities like helpfulness, safety, or coherence by using natural language understanding. This approach is best suited for qualitative analysis where nuanced understanding is required, such as evaluating the helpfulness of responses, detecting subtle hallucinations, or assessing the quality of creative content.

```python
# An LLM grader that evaluates helpfulness of responses
helpfulness_grader = HelpfulnessGrader(model=OpenAIChatModel("qwen3-32b"))
```

### Evaluation Modes

Based on the evaluation purpose, you can choose the appropriate evaluation mode. Each mode produces specific output types:

- **Pointwise evaluation**: Assesses individual samples independently, generating a score for each input-output pair. This approach is suitable for quantitative assessment of individual responses, producing a [GraderScore](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/graders/schema.py) for each sample.

- **Listwise evaluation**: Ranks multiple samples relative to each other, comparing several responses to the same query. This approach generates relative rankings rather than absolute scores, producing a [GraderRank](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/graders/schema.py) that indicates the relative quality of responses.

### Understanding Results

Graders return different result types depending on their mode:

#### GraderScore (Pointwise Mode)

For evaluating individual responses, graders return a `GraderScore` object:

| Field | Description |
|-------|-------------|
| `score` | Numerical score (e.g., 0-1 or 1-5 scale) |
| `reason` | Explanation for the score |
| `name` | Name of the grader used |
| `metadata` | Additional details (e.g., threshold settings) |

#### GraderRank (Listwise Mode)

For ranking multiple responses, graders return a `GraderRank` object:

| Field | Description |
|-------|-------------|
| `rank` | Ranking of responses (e.g., [1, 3, 2] means 1st is best, 3rd is second, 2nd is worst) |
| `reason` | Explanation for the ranking |
| `name` | Name of the grader used |
| `metadata` | Additional ranking details |

Both output types maintain consistency across different grader implementation, making it easy to combine and analyze results.

### Built-in Predefined Graders

OpenJudge comes with a rich collection of pre-built graders across various domains including common, agent, code, format, multimodal, math, and text evaluations. These graders are ready to use and cover most common evaluation scenarios.

For a comprehensive list and detailed usage of the predefined graders, please refer to the [Built-in Graders documentation](../built_in_graders/overview.md).

### Creating Custom Graders

While OpenJudge provides many built-in graders, you'll often need to create custom graders for your specific use cases. The system supports three main approaches to grader creation:

- **Custom implementation**: Implementing your own graders using Code-Based or LLM-Based approaches
- **Automated generation**: Using the generator module to automatically create graders from data
- **Model training**: Training specialized models for evaluation (using supervised or reinforcement learning)

For detailed instructions on creating custom graders, please refer to the [Building Custom Graders documentation](../building_graders/create_custom_graders.md).

## How Runners Use Graders

The [GradingRunner](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/runner/grading_runner.py) is the central execution engine of OpenJudge that orchestrates the evaluation process across multiple graders. It acts as the conductor of an orchestra, coordinating all the different graders to create a harmonious evaluation process. The GradingRunner is specifically designed to serve Graders by providing the infrastructure they need to operate efficiently and effectively. It serves Graders by providing execution orchestration, data mapping services, concurrency management, result aggregation, and resource optimization to execute graders concurrently and maximize throughput.

To better understand how Runners use Graders, let's look at a complete configuration example:

```python
from openjudge.runner.grading_runner import GradingRunner
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.graders.common.helpfulness import HelpfulnessGrader
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
graders = {
    "helpfulness": {
        "grader": HelpfulnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"question": "query", "answer": "response"}
    },
    "relevance": {
        "grader": RelevanceGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"q": "query", "a": "response", "ref": "reference_answer"}
    }
}

# Configure aggregators to combine results
aggregators = [
    WeightedSumAggregator(weights={"helpfulness": 0.6, "relevance": 0.4})
]

# Run evaluation with concurrency control
runner = GradingRunner(
    graders,
    aggregators=aggregators,
    max_concurrency=5
)
results = await runner.arun(data)
```

This example demonstrates several key concepts that are essential to understanding how Runners use Graders:

#### Data Mapping

The GradingRunner's mapper functionality allows you to transform your data fields to match the parameter names expected by your graders. Since your input data may not have the exact field names that your graders expect, mappers provide a way to map between your data structure and the grader's expected inputs.

In the example above:
- The HelpfulnessGrader expects inputs named "question" and "answer", but our data has fields named "query" and "response", so the mapper `{"question": "query", "answer": "response"}` connects them.
- The RelevanceGrader expects inputs named "q", "a", and "ref", so the mapper `{"q": "query", "a": "response", "ref": "reference_answer"}` maps our data fields to the grader's expected inputs.

Types of mappers include:
- Dictionary mappers for simple key-value mappings (e.g., `{"question": "query", "answer": "response"}`)
- Callable mappers for custom functions that transform data in more complex ways

#### Aggregation Configuration

After running multiple graders, you might want to combine their results into a single score. The [aggregator submodule](https://github.com/agentscope-ai/OpenJudge/tree/main/openjudge/runner/aggregator/) provides components that take multiple grader results and combine them into a unified result:

- **WeightedSumAggregator**: Combining results using weighted averages. In our example, we assign 60% weight to helpfulness and 40% to relevance: `WeightedSumAggregator(weights={"helpfulness": 0.6, "relevance": 0.4})`
- **MaxAggregator**: Taking the maximum score among all graders
- **MinAggregator**: Taking the minimum score among all graders

These aggregators allow you to create composite scores that reflect multiple evaluation dimensions, making it easier to compare overall performance across different models or configurations.

#### Concurrency Control

The GradingRunner is designed for high-performance evaluation by managing execution concurrency:

- **Multi-Grader concurrency**: Multiple different graders execute concurrently for each data item, improving evaluation speed
- **Data concurrency**: Multiple data items are processed concurrently across all graders
- **Concurrency limits**: The `max_concurrency` parameter controls the maximum number of concurrent operations to prevent system overload. In our example, `max_concurrency=5` limits the system to processing 5 items simultaneously.

Concurrency control enables efficient processing of large datasets while maintaining system stability.

## How to Analyze Grader

After running evaluations with the **GradingRunner**, you can use the [analyzer module](https://github.com/agentscope-ai/OpenJudge/tree/main/openjudge/analyzer/) to process the results and gain deeper insights. Analyzers are optional components that help you understand your evaluation results better and assess the quality of your graders.

Types of analyzers include:
- **Statistical analyzers**: Compute statistics on evaluation results (e.g., [DistributionAnalyzer](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/analyzer/statistical/distribution_analyzer.py)) to understand score distributions and identify potential issues with grader consistency
- **Validation analyzers**: Compare evaluation results with reference labels (e.g., [AccuracyAnalyzer](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/analyzer/validation/accuracy_analyzer.py), [F1ScoreAnalyzer](https://github.com/agentscope-ai/OpenJudge/blob/main/openjudge/analyzer/validation/f1_score_analyzer.py)) to measure how well your graders correlate with known ground truth

These analyzers help you:
- Evaluate the effectiveness of your graders by comparing their outputs to known standards
- Identify potential bias or inconsistency in your evaluation process
- Understand the statistical properties of your grader scores
- Validate that your graders are measuring what you intend them to measure

## Next Steps

- [Built-in Graders](../built_in_graders/overview.md) — Explore 50+ pre-built graders for various evaluation scenarios
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build specialized graders for your domain
- [Validate Graders](../validating_graders/overview.md) — Ensure your graders produce reliable results