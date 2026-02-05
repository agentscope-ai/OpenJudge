# Build Reward for Training

Reinforcement Learning from Human Feedback (RLHF) and post-training optimization rely on reward signals to guide model behavior. This guide walks you through building rewards for a real-world task: **training a customer support chatbot**. You'll start with a single grader to evaluate one quality dimension, then learn to combine multiple graders for a comprehensive reward signal.

!!! note "Additional Resources"
    For detailed grader documentation, see [Built-in Graders](../built_in_graders/overview.md).


## The Task: Customer Support Chatbot

Imagine you're building a chatbot to handle refund policy questions for an e-commerce platform. Your chatbot needs to provide helpful, accurate answers while avoiding inappropriate content. In this guide, we'll build a reward system that evaluates model responses on these dimensions and generates training signals for RLHF or DPO.

The training process requires computing a reward score for each model response. This reward will guide the model to generate better responses through reinforcement learning. We'll start simple with a single quality check, then progressively add more evaluation criteria to create a robust reward signal.


## Start Simple: Evaluate Relevance

Before worrying about multiple quality dimensions, let's focus on one fundamental requirement: responses should be relevant to user questions. A chatbot that answers "The sky is blue" to "What is your refund policy?" is useless, regardless of other qualities.

We'll use the `RelevanceGrader` to evaluate whether model responses address the user's question. This grader uses an LLM to assess topical relevance and returns a score from 1 (completely off-topic) to 5 (directly addresses the question).

First, prepare a small dataset with queries and responses. Each entry needs at least two fields: `query` (user question) and `response` (model answer):

```python
dataset = [
    {
        "query": "What is your refund policy?",
        "response": "We offer full refunds within 30 days of purchase."
    },
    {
        "query": "How do I reset my password?",
        "response": "Go to Settings > Security > Reset Password."
    },
]
```

Now run the relevance grader on this dataset. The `GradingRunner` handles execution and returns relevance scores for each response:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import RelevanceGrader
from openjudge.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    # Initialize the grading model
    model = OpenAIChatModel(model="qwen3-32b")

    # Configure the relevance grader
    grader_configs = {
        "relevance": GraderConfig(grader=RelevanceGrader(model=model))
    }

    # Create runner and execute
    runner = GradingRunner(
        grader_configs=grader_configs,
        max_concurrency=32,
        show_progress=True
    )

    results = await runner.arun(dataset)

    # Print results
    for i, result in enumerate(results["relevance"]):
        print(f"Sample {i}: relevance={result.score}/5")
        print(f"  Reason: {result.reason}\n")

asyncio.run(main())
```

This gives you a basic reward signal based on relevance. For simple use cases, this single-dimension reward might be sufficient. But for production systems, you typically need to evaluate multiple quality dimensions simultaneously.

## Advanced Evaluation: Using Evaluation Strategies

For more robust evaluation, you can use evaluation strategies that control how the grader executes its assessment. Strategies determine how many times an evaluation is called, in what order, and how results are aggregated.

For example, you can use a voting strategy to run the evaluation multiple times and aggregate results using voting mechanisms:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import RelevanceGrader
from openjudge.evaluation_strategy.voting_evaluation_strategy import VotingEvaluationStrategy
from openjudge.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    # Initialize the grading model
    model = OpenAIChatModel(model="qwen3-32b")

    # Configure the relevance grader with a voting strategy
    voting_strategy = VotingEvaluationStrategy(num_votes=3)
    grader_configs = {
        "relevance": GraderConfig(
            grader=RelevanceGrader(model=model, strategy=voting_strategy)
        )
    }

    # Create runner and execute
    runner = GradingRunner(
        grader_configs=grader_configs,
        max_concurrency=32,
        show_progress=True
    )

    results = await runner.arun(dataset)

    # Print results
    for i, result in enumerate(results["relevance"]):
        print(f"Sample {i}: relevance={result.score}/5")
        print(f"  Reason: {result.reason}\n")

asyncio.run(main())
```

Other strategies include:
- **AverageEvaluationStrategy**: Averages numerical results across multiple runs to reduce noise
- **DirectEvaluationStrategy**: Executes the evaluation once (this is the default when no strategy is specified)

## Add More Dimensions: Build Composite Rewards

Relevance alone isn't enough for a production chatbot. You also need to ensure responses are safe (no offensive content) and accurate (factually correct). This is where composite rewards come in—combining multiple graders to evaluate different quality dimensions simultaneously.

For our customer support chatbot, we'll add two more graders: `HarmfulnessGrader` to detect inappropriate content and `CorrectnessGrader` to verify factual accuracy against ground truth answers. Each grader evaluates one dimension and returns a score from 1 to 5.

Before running multiple graders, we need to update our dataset to include ground truth answers for accuracy checking. We'll add a `ground_truth` field that contains the correct answer for each query:

```python
dataset = [
    {
        "query": "What is your refund policy?",
        "response": "We offer full refunds within 30 days of purchase.",
        "ground_truth": "Full refunds within 30 days."
    },
    {
        "query": "How do I reset my password?",
        "response": "Go to Settings > Security > Reset Password.",
        "ground_truth": "Settings > Security > Reset Password"
    },
]
```

Notice that our data uses `ground_truth` as the field name, but `CorrectnessGrader` expects a field called `reference_response`. When field names don't match, use the `mapper` parameter to map fields. The mapper extracts and renames fields from your data to match the grader's expected inputs:

```python
# Map fields: extract query, response, and rename ground_truth to reference_response
mapper = {
    "query": "query",
    "response": "response",
    "reference_response": "ground_truth"
}
```

Now we can configure all three graders and combine their scores into a single reward. The `WeightedSumAggregator` computes a weighted average of individual grader scores, letting you control how much each quality dimension contributes to the final reward:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import HarmfulnessGrader, RelevanceGrader, CorrectnessGrader
from openjudge.runner.grading_runner import GradingRunner, GraderConfig
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

async def main():
    model = OpenAIChatModel(model="qwen3-32b")

    # Configure three graders for different quality dimensions
    grader_configs = {
        "harmfulness": GraderConfig(
            grader=HarmfulnessGrader(model=model),
            mapper={"query": "query", "response": "response"}
        ),
        "relevance": GraderConfig(
            grader=RelevanceGrader(model=model),
            mapper={"query": "query", "response": "response"}
        ),
        "correctness": GraderConfig(
            grader=CorrectnessGrader(model=model),
            mapper={
                "query": "query",
                "response": "response",
                "reference_response": "ground_truth"
            }
        ),
    }

    # Combine scores using weighted average
    aggregator = WeightedSumAggregator(
        name="training_reward",
        weights={
            "harmfulness": 0.4,   # Safety matters most
            "relevance": 0.3,     # Then relevance
            "correctness": 0.3,   # Then accuracy
        }
    )

    runner = GradingRunner(
        grader_configs=grader_configs,
        aggregators=aggregator,
        max_concurrency=32,
        show_progress=True
    )

    results = await runner.arun(dataset)

    # Print aggregated rewards
    for i, reward in enumerate(results["training_reward"]):
        print(f"Sample {i}: reward={reward.score:.2f}")

asyncio.run(main())
```

The aggregator computes a weighted sum of the three grader scores. For example, if a response gets harmfulness=5.0, relevance=4.0, and correctness=3.0, the final reward would be `0.4×5.0 + 0.3×4.0 + 0.3×3.0 = 4.1`. Weight selection depends on your priorities—for safety-critical applications like customer support, you might assign higher weight to harmfulness (0.5+), while knowledge-intensive tasks benefit from emphasizing correctness (0.4+).

You can also inspect individual grader scores to understand how each dimension contributes to the final reward. This is useful for debugging why certain responses receive low rewards:

```python
results = await runner.arun(dataset)

for i in range(len(dataset)):
    print(f"\nSample {i}:")
    print(f"  Harmfulness: {results['harmfulness'][i].score}/5")
    print(f"  Relevance: {results['relevance'][i].score}/5")
    print(f"  Correctness: {results['correctness'][i].score}/5")
    print(f"  → Final Reward: {results['training_reward'][i].score:.2f}")
```

If you don't specify weights, the aggregator automatically assigns equal weights to all graders. For more complex scenarios, you can write custom aggregation logic—for instance, using the minimum score across all graders to create a conservative reward signal that penalizes any dimension that falls short.


## Putting It All Together

Here's the complete workflow for building a composite reward signal for our customer support chatbot. This example shows how to prepare data, configure multiple graders, aggregate scores, and extract training rewards:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import HarmfulnessGrader, RelevanceGrader, CorrectnessGrader
from openjudge.runner.grading_runner import GradingRunner, GraderConfig
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

async def main():
    model = OpenAIChatModel(model="qwen3-32b")

    # Prepare training data
    dataset = [
        {
            "query": "What is your refund policy?",
            "response": "We offer full refunds within 30 days of purchase.",
            "ground_truth": "Full refunds within 30 days."
        },
        {
            "query": "How do I reset my password?",
            "response": "Go to Settings > Security > Reset Password.",
            "ground_truth": "Settings > Security > Reset Password"
        },
    ]

    # Configure graders
    grader_configs = {
        "harmfulness": GraderConfig(
            grader=HarmfulnessGrader(model=model),
            mapper={"query": "query", "response": "response"}
        ),
        "relevance": GraderConfig(
            grader=RelevanceGrader(model=model),
            mapper={"query": "query", "response": "response"}
        ),
        "correctness": GraderConfig(
            grader=CorrectnessGrader(model=model),
            mapper={
                "query": "query",
                "response": "response",
                "reference_response": "ground_truth"
            }
        ),
    }

    # Configure aggregation
    aggregator = WeightedSumAggregator(
        name="training_reward",
        weights={"harmfulness": 0.4, "relevance": 0.3, "correctness": 0.3},
    )

    # Run evaluation
    runner = GradingRunner(
        grader_configs=grader_configs,
        aggregators=aggregator,
        max_concurrency=32,
        show_progress=True
    )

    results = await runner.arun(dataset)

    # Print detailed results
    print("=== Training Rewards ===")
    for i, reward_result in enumerate(results["training_reward"]):
        print(f"\nSample {i}: {dataset[i]['query']}")
        print(f"  Harmfulness: {results['harmfulness'][i].score}/5")
        print(f"  Relevance: {results['relevance'][i].score}/5")
        print(f"  Correctness: {results['correctness'][i].score}/5")
        print(f"  → Final Reward: {reward_result.score:.2f}")

asyncio.run(main())
```

Running this code evaluates both responses across three quality dimensions and produces a training reward for each. These rewards can then feed into RLHF or DPO algorithms to optimize your chatbot. The output shows individual dimension scores alongside the final aggregated reward, helping you understand what drives the training signal.

You now have a foundation for building composite rewards. Start with a single grader to validate your setup, then progressively add more dimensions as needed. The key is choosing graders that align with your application's requirements and weighting them appropriately based on what matters most for your use case.


## Explore More Graders

Beyond the three graders used in this tutorial, OpenJudge provides 50+ built-in graders covering various quality dimensions. Different applications require different evaluation criteria, so it's worth exploring what's available.

For text-based applications, you might need graders that check for hallucinations, measure response conciseness, or validate output format. The [General Graders](../built_in_graders/general.md) and [Format Graders](../built_in_graders/format.md) documentation covers these use cases. If you're building AI agents that use tools or follow multi-step reasoning, check out [Agent Graders](../built_in_graders/agent_graders.md) for evaluating tool selection accuracy, action alignment, and trajectory quality.

When built-in graders don't cover your specific requirements, you can create custom graders tailored to your domain. See [Create Custom Graders](../building_graders/create_custom_graders.md) for guidance on building evaluators that understand your application's unique constraints and quality standards.


## Next Steps

- [Built-in Graders Overview](../built_in_graders/overview.md) — Browse all available graders organized by category
- [Run Grading Tasks](../running_graders/run_tasks.md) — Explore parallel execution and result analysis
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific graders for specialized use cases
