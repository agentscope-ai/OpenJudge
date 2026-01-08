# Building External Evaluation Pipelines with OpenJudge for LangSmith

This guide walks you through building robust external evaluation pipelines for LangSmith using OpenJudge. By connecting these two tools, you can harness OpenJudge's 50+ built-in graders to deliver thorough automated quality assessment of your LLM applications within the LangSmith platform.

## Overview

### Why Implement External Evaluation Pipelines?

Although LangSmith offers native evaluation features, external evaluation pipelines provide additional benefits:

- **Adaptable Execution**: Initiate evaluations anytime, separate from application execution cycles
- **Comprehensive Assessment**: OpenJudge delivers 50+ graders addressing quality, safety, formatting, agent behaviors, and beyond
- **Customizability**: Seamlessly incorporate custom evaluation logic tailored to specific business requirements
- **Scalable Processing**: Efficiently handle large volumes of historical runs with support for scheduled tasks and incremental assessment

### Integration Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   LangSmith     │      │    OpenJudge    │      │   LangSmith     │
│(Dataset+App)    │─────▶│    (Graders)    │─────▶│    (Scores)     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        ▲
        │   dataset & app        │   evaluate()           │   feedback/score
        └────────────────────────┴────────────────────────┘
```

The overall workflow consists of three stages:

1. **Dataset & Application Setup**: Establish datasets in LangSmith and specify the application to be assessed
2. **Execute Evaluation**: Deploy OpenJudge graders as custom evaluators inside LangSmith's evaluation framework
3. **Save Results**: LangSmith records the evaluation scores as feedback

## Prerequisites

### Install Dependencies

```bash
pip install py-openjudge langsmith python-dotenv
```

### Configure Environment Variables

```bash
# LangSmith authentication
export LANGSMITH_API_KEY="ls-your-api-key"

# OpenAI API configuration (required for LLM-based graders)
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
```

### Initialize Clients

```python
import os
from langsmith import Client

# Initialize LangSmith client
client = Client()

# Verify connection
try:
    client.info()  # Check if the client is properly authenticated
    print("LangSmith authentication successful")
except Exception as e:
    print(f"LangSmith authentication failed: {e}")
```

## Step 1: Create or Select a Dataset

We require a [Dataset](https://docs.smith.langchain.com/langsmith/evaluation-concepts#datasets) to assess our application. Our dataset will comprise labeled [examples](https://docs.smith.langchain.com/langsmith/evaluation-concepts#examples) containing questions and expected answers.

```python
from langsmith import Client

client = Client()

# Create examples with inputs and expected outputs
examples = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"expected_answer": "Paris"}
    },
    {
        "inputs": {"question": "How many planets are in our solar system?"},
        "outputs": {"expected_answer": "8"}
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "outputs": {"expected_answer": "William Shakespeare"}
    }
]

# Create the dataset
dataset = client.create_dataset(dataset_name="QA Evaluation Dataset")
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)
```

For additional details on datasets, consult the [Manage datasets](https://docs.smith.langchain.com/langsmith/manage-datasets) page.

## Step 2: Define an Application

Initially, we need an application for evaluation. Here's an example QA application that we aim to evaluate:

```python
from open_judge.models.openai_chat_model import OpenAIChatModel
import asyncio

def qa_application(inputs: dict) -> dict:
    """
    The target application to be evaluated.

    Args:
        inputs: Dictionary containing input data

    Returns:
        Dictionary containing the application output
    """
    model = OpenAIChatModel(model="gpt-3.5-turbo")
    response = asyncio.run(model.achat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": inputs["question"]}
    ]))
    return {"answer": response.content}
```

We've optionally enabled tracing to capture the inputs and outputs of each step in the pipeline. To understand how to annotate your code for tracing, please refer to the [LangSmith documentation](https://docs.smith.langchain.com/).

## Step 3: Build a Custom Evaluator from OpenJudge Graders

### Selecting the Appropriate Grader

Choose the suitable OpenJudge grader based on your evaluation requirements:

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

### Option 1: Single Grader (Quick Start)

The most straightforward approach is to perform evaluation using a single grader:

```python
from typing import Callable, Dict, Any, Union, Awaitable
from open_judge.graders.base_grader import BaseGrader
from open_judge.graders.schema import GraderResult, GraderScore, GraderRank, GraderError
from open_judge.utils.mapping import parse_data_with_mapper

def create_langsmith_evaluator(grader: BaseGrader, mapper: dict | None = None):
    """
    Create a LangSmith-compatible evaluator from an OpenJudge grader.

    Args:
        grader: An OpenJudge grader instance
        mapper: A dictionary mapping source keys to target keys for data transformation

    Returns:
        A LangSmith-compatible evaluator function
    """
    def langsmith_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """
        LangSmith evaluator function that processes input, output and reference data.

        Args:
            inputs: The inputs from LangSmith example
            outputs: The actual outputs from LangSmith run
            reference_outputs: The expected outputs from LangSmith example

        Returns:
            A dictionary containing the evaluation results with score and reasoning
        """
        try:
            # Prepare data for evaluation
            data = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}

            # Parse and map the data using the mapper
            mapped_data = parse_data_with_mapper(data, mapper)

            # Execute OpenJudge evaluation with the mapped data
            result: GraderResult = asyncio.run(grader.aevaluate(**mapped_data))

            # Convert OpenJudge result to LangSmith format
            if isinstance(result, GraderScore):
                return {
                    "key": grader.name,  # The feedback key for LangSmith
                    "score": result.score,
                    "comment": getattr(result, 'reason', '')
                }
            elif isinstance(result, GraderError):
                return {
                    "key": grader.name,
                    "score": 0.0,
                    "comment": f"Error: {result.error}"
                }
            else:
                return {
                    "key": grader.name,
                    "score": 0.0,
                    "comment": "Unknown result type"
                }
        except Exception as e:
            # Handle any unexpected errors during evaluation
            return {
                "key": grader.name,
                "score": 0.0,
                "comment": f"Evaluation failed: {str(e)}"
            }

    return langsmith_evaluator

# Example usage with specific graders and mappers
from open_judge.graders.common.relevance import RelevanceGrader
from open_judge.graders.common.correctness import CorrectnessGrader
from open_judge.models.openai_chat_model import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})

# Define mappers for each grader - mapping LangSmith data format to OpenJudge format
relevance_mapper = {
    "query": "inputs.question",
    "response": "outputs.answer",
}

correctness_mapper = {
    "query": "inputs.question",
    "response": "outputs.answer",
    "reference_response": "reference_outputs.expected_answer"
}

graders = [
    ("relevance", RelevanceGrader(model=model), relevance_mapper),
    ("correctness", CorrectnessGrader(model=model), correctness_mapper)
]

# Convert to LangSmith evaluators
langsmith_evaluators = [
    create_langsmith_evaluator(grader, mapper)
    for _, grader, mapper in graders
]

# Run evaluation with individual graders
from langsmith.evaluation import evaluate

experiment_results = evaluate(
    qa_application,  # Your LLM application or chain
    data=dataset.name,  # Dataset in LangSmith
    evaluators=langsmith_evaluators,
    experiment_prefix="open_judge_individual_graders",
    description="Evaluate QA application using OpenJudge's individual graders with LangSmith integration",
    max_concurrency=4
)
```

### Option 2: Batch Evaluation with GradingRunner (Recommended)

For more complex situations involving multiple graders, OpenJudge's GradingRunner delivers efficient batch processing capabilities with integrated mapper support:

```python
from open_judge.runner.grading_runner import GradingRunner
from open_judge.graders.common.correctness import CorrectnessGrader
from open_judge.graders.common.relevance import RelevanceGrader
from open_judge.models.openai_chat_model import OpenAIChatModel
import asyncio

class LangSmithBatchEvaluator:
    """Batch evaluator that combines multiple OpenJudge graders for LangSmith integration"""

    def __init__(self, model=None, mapper: dict | None = None):
        """
        Initialize the batch evaluator with a GradingRunner.

        Args:
            model: Model instance for graders that require it
            mapper: A dictionary mapping source keys to target keys for data transformation
        """
        if model is None:
            model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})

        # Enhanced grader configuration with diverse evaluation dimensions
        grader_configs = {
            "relevance": (RelevanceGrader(model=model), mapper),
            "correctness": (CorrectnessGrader(model=model), mapper),
        }

        # Configure the runner with comprehensive grader suite
        self.runner = GradingRunner(
            grader_configs=grader_configs,
            max_concurrency=8,  # Slightly reduced for more stable LLM-based evaluations
            timeout=30
        )

    def __call__(self, inputs: dict, outputs: dict, reference_outputs: dict) -> list:
        """
        LangSmith batch evaluator function.

        Args:
            inputs: The inputs from LangSmith example
            outputs: The actual outputs from LangSmith run
            reference_outputs: The expected outputs from LangSmith example

        Returns:
            A list of dictionaries containing results from all graders
        """
        try:
            # Prepare data for batch processing
            data = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}

            # Execute batch evaluation using OpenJudge runner
            batch_results = asyncio.run(self.runner.arun([data]))

            # Convert results to LangSmith format
            formatted_results = []
            for grader_name, grader_results in batch_results.items():
                if grader_results:  # Check if results exist
                    result = grader_results[0]  # We only have one sample
                    if isinstance(result, GraderScore):
                        formatted_results.append({
                            "key": grader_name,
                            "score": result.score,
                            "comment": getattr(result, "reason", "")
                        })
                    elif isinstance(result, GraderError):
                        formatted_results.append({
                            "key": grader_name,
                            "score": 0.0,
                            "comment": f"Error: {result.error}"
                        })
                    else:
                        formatted_results.append({
                            "key": grader_name,
                            "score": 0.0,
                            "comment": "Unknown result type"
                        })

            return formatted_results

        except Exception as e:
            # Handle any errors during batch evaluation
            return [{
                "key": "batch_evaluation_error",
                "score": 0.0,
                "comment": f"Batch evaluation failed: {str(e)}"
            }]

# Define comprehensive mapper for the batch evaluator - mapping LangSmith data format to OpenJudge format
mapper = {
    "query": "inputs.question",
    "response": "outputs.answer",
    "reference_response": "reference_outputs.expected_answer"
    # The instruction_following grader uses a simplified mapper defined in its config
}

# Usage example
batch_evaluator = LangSmithBatchEvaluator(mapper=mapper)

# Run evaluation with GradingRunner
from langsmith.evaluation import evaluate

experiment_results = evaluate(
    qa_application,
    data=dataset.name,
    evaluators=[batch_evaluator],  # Single batch evaluator handles multiple graders
    experiment_prefix="open_judge_batch_evaluation",
    description="Evaluating QA application with OpenJudge GradingRunner",
    max_concurrency=4
)
```

## Step 4: Explore the results

Each invocation of [evaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate) creates an Experiment which can be viewed in the LangSmith UI or queried via the SDK. Evaluation scores are stored against each actual output as feedback.

The results can be accessed programmatically:

```python
# Convert experiment results to pandas DataFrame for analysis
df = experiment_results.to_pandas()

# Access specific metrics
print("Average relevance score:", df["relevance"].mean())
print("Average correctness score:", df["correctness"].mean())

# Analyze results
for _, row in df.iterrows():
    print(f"Input: {row['inputs']}")
    print(f"Output: {row['outputs']}")
    print(f"Relevance: {row['relevance']}")
    print(f"Correctness: {row['correctness']}")
    print("---")
```


## Additional Learning Materials

- **Comprehensive Grading Toolkit**: Explore [OpenJudge's 50+ Built-in Graders](../built_in_graders/overview.md) for immediate use cases
- **Extensibility Guide**: Learn how to [Create Custom Graders](../building_graders/create_custom_graders.md) for domain-specific evaluation needs
- **Technical Reference**: Dive into the complete [OpenJudge Graders API Documentation](../../openjudge/graders/)
- **Evaluation Fundamentals**: Understand core concepts in the [LangSmith Evaluation Guide](https://docs.langchain.com/langsmith/evaluate-llm-application)
- **Data Management**: Master dataset creation and maintenance with [LangSmith Dataset Management](https://docs.smith.langchain.com/langsmith/manage-datasets)
- **Conceptual Framework**: Study evaluation principles in the [LangSmith Evaluation Concepts](https://docs.smith.langchain.com/langsmith/evaluation-concepts) documentation