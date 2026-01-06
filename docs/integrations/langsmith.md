# LangSmith Integration Guide

## Overview

LangSmith is an LLM application development and monitoring platform provided by LangChain. It adopts a "contract-based" integration model, treating evaluators as part of the experimental pipeline. This means LangSmith deeply integrates into the runtime flow, requiring evaluators to be packaged as Python Callables that accept specific inputs and return specific outputs. OpenJudge can be easily integrated with LangSmith by wrapping OpenJudge graders as LangSmith-compatible evaluators, enabling seamless evaluation of LLM applications within the LangSmith ecosystem.

## Prerequisites

### Installation

First, install the required dependencies:

```bash
pip install langsmith open_judge python-dotenv
```

### Environment Configuration

Create a `.env` file with your API keys:

```
LANGSMITH_API_KEY=your_langsmith_api_key
OPENAI_API_KEY=your_openai_api_key  # if using OpenAI models
```

Then load the environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
```

## Define an application

First we need an application to evaluate. Here's an example QA application that we want to evaluate:

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
    model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})
    response = asyncio.run(model.achat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": inputs["question"]}
    ]))
    return {"answer": response.content}
```

We've optionally enabled tracing to capture the inputs and outputs of each step in the pipeline. To understand how to annotate your code for tracing, please refer to the [LangSmith documentation](https://docs.smith.langchain.com/).

## Create or select a dataset

We need a [Dataset](https://docs.smith.langchain.com/langsmith/evaluation-concepts#datasets) to evaluate our application on. Our dataset will contain labeled [examples](https://docs.smith.langchain.com/langsmith/evaluation-concepts#examples) of questions and expected answers.

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
    inputs=[ex["inputs"] for ex in examples],
    outputs=[ex["outputs"] for ex in examples]
)
```

For more details on datasets, refer to the [Manage datasets](https://docs.smith.langchain.com/langsmith/manage-datasets) page.

## Define an evaluator

[Evaluators](https://docs.smith.langchain.com/langsmith/evaluation-concepts#evaluators) are functions for scoring your application's outputs. They take in the example inputs, actual outputs, and, when present, the reference outputs. In this section, we'll show how to integrate OpenJudge graders with LangSmith in two ways: using individual graders and using the GradingRunner.

### Using Individual OpenJudge Graders

The following function wraps an OpenJudge grader to make it compatible with LangSmith's evaluation interface. This version includes mapper support for data transformation:

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
            data = {
                "inputs": inputs,
                "outputs": outputs,
                "reference_outputs": reference_outputs
            }

            # Parse and map the data using the mapper
            mapped_data = parse_data_with_mapper(data, mapper)

            # Execute OpenJudge evaluation with the mapped data
            result: GraderResult = asyncio.run(grader.aevaluate(**evaluation_data))

            # Convert OpenJudge result to LangSmith format
            if isinstance(result, GraderScore):
                return {
                    "key": grader.name,  # The feedback key for LangSmith
                    "score": result.score,
                    "comment": getattr(result, 'reason', '')
                }
            elif isinstance(result, GraderRank):
                return {
                    "key": grader.name,
                    "score": getattr(result, 'rank', 0),
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
```

### Using OpenJudge GradingRunner

For more complex scenarios involving multiple graders, OpenJudge's GradingRunner provides efficient batch processing capabilities with built-in mapper support:

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

        # Define grader configs with their respective mappers
        grader_configs = {
            "relevance": (RelevanceGrader(model=model), mapper),
            "correctness": (CorrectnessGrader(model=model), mapper),
        }

        # Configure the runner with multiple graders and their mappers
        self.runner = GradingRunner(
            grader_configs=grader_configs,
            max_concurrency=10
        )

        # Store the top-level mapper for input preprocessing
        self.mapper = mapper or {}

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
            data = {
                "inputs": inputs,
                "outputs": outputs,
                "reference_outputs": reference_outputs
            }

            # Wrap in list for batch processing
            evaluation_data = [final_inputs]

            # Execute batch evaluation using OpenJudge runner
            batch_results = asyncio.run(self.runner.arun(evaluation_data))

            # Convert results to LangSmith format
            formatted_results = []
            for grader_name, grader_results in batch_results.items():
                if grader_results:  # Check if results exist
                    result = grader_results[0]  # We only have one sample
                    if isinstance(result, GraderScore):
                        formatted_results.append({
                            "key": grader_name,
                            "score": result.score,
                            "comment": getattr(result, 'reason', '')
                        })
                    elif isinstance(result, GraderRank):
                        formatted_results.append({
                            "key": grader_name,
                            "score": getattr(result, 'rank', 0),
                            "comment": getattr(result, 'reason', '')
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

# Define mapper for the batch evaluator - mapping LangSmith data format to OpenJudge format
mapper = {
    "query": "inputs.question",
    "response": "outputs.answer",
    "reference_response": "reference_outputs.expected_answer"
}

# Usage example
batch_evaluator = LangSmithBatchEvaluator(mapper=mapper)
```

## Run the evaluation

We'll use the [evaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate)/[aevaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._arunner.aevaluate) methods to run the evaluation. The key arguments are:
- a target function that takes an input dictionary and returns an output dictionary
- data - the name OR UUID of the LangSmith dataset to evaluate on, or an iterator of examples
- evaluators - a list of evaluators to score the outputs of the function

### Using Individual Graders

```python
from langsmith.evaluation import evaluate

# Run evaluation with individual graders
experiment_results = evaluate(
    qa_application,  # Your LLM application or chain
    data=dataset.name,  # Dataset in LangSmith
    evaluators=langsmith_evaluators,
    experiment_prefix="open_judge_individual_graders",
    description="Evaluating QA application with OpenJudge individual graders",
    max_concurrency=4
)
```

### Using GradingRunner

```python
# Run evaluation with GradingRunner
experiment_results = evaluate(
    qa_application,
    data=dataset.name,
    evaluators=[batch_evaluator],  # Single batch evaluator handles multiple graders
    experiment_prefix="open_judge_batch_evaluation",
    description="Evaluating QA application with OpenJudge GradingRunner",
    max_concurrency=4
)
```

### Asynchronous Evaluation

For large evaluation jobs, use the asynchronous version of [evaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate), which is [aevaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._arunner.aevaluate):

```python
from langsmith.evaluation import aevaluate

async def run_async_evaluation():
    """Run evaluation asynchronously for better performance with large datasets."""
    async for run in aevaluate(
        qa_application,
        data=dataset.name,
        evaluators=langsmith_evaluators,
        max_concurrency=4
    ):
        print(f"Completed run: {run}")
```

## Explore the results

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

## Related source

- [OpenJudge GradingRunner Documentation](../../openjudge/runner/grading_runner.py)
- [LangSmith Evaluation Concepts](https://docs.smith.langchain.com/langsmith/evaluation-concepts)
- [OpenJudge Graders Documentation](../../openjudge/graders/)
- [LangSmith Dataset Management](https://docs.smith.langchain.com/langsmith/manage-datasets)
- [LangSmith Evaluation Guide](https://docs.langchain.com/langsmith/evaluate-llm-application)