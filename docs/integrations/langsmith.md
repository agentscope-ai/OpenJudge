# LangSmith Integration

This guide walks you through building robust external evaluation pipelines for LangSmith using OpenJudge. By connecting these two tools, you can harness OpenJudge's 50+ built-in graders to deliver thorough automated quality assessment of your LLM applications within the LangSmith platform.

## Overview

Although LangSmith offers native evaluation features, external evaluation pipelines provide additional benefits:

- **Adaptable Execution**: Initiate evaluations anytime, separate from application execution cycles
- **Comprehensive Assessment**: OpenJudge delivers 50+ graders addressing quality, safety, formatting, agent behaviors, and beyond
- **Customizability**: Seamlessly incorporate custom evaluation logic tailored to specific business requirements
- **Scalable Processing**: Efficiently handle large volumes of historical runs with support for scheduled tasks and incremental assessment

The integration follows a simple three-step flow:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   LangSmith     │      │    OpenJudge    │      │   LangSmith     │
│(Dataset+App)    │─────▶│    (Graders)    │─────▶│    (Scores)     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

1. **Prepare Dataset & App** — Create a dataset in LangSmith and define your application
2. **Build Evaluators** — Wrap OpenJudge graders as LangSmith-compatible evaluators
3. **Run & View Results** — Execute evaluation and explore results in LangSmith UI

## Prerequisites

!!! warning "Required Configuration"
    Make sure to set all required environment variables before running the code.

**Install dependencies:**

```bash
pip install py-openjudge langsmith python-dotenv
```

**Configure environment variables:**

```bash
# LangSmith authentication
export LANGSMITH_API_KEY="ls-your-api-key"

# OpenAI API configuration (required for LLM-based graders)
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
```

**Initialize the LangSmith client:**

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

## Integration Steps

### Step 1: Create a Dataset

We require a [Dataset](https://docs.smith.langchain.com/langsmith/evaluation-concepts#datasets) to assess our application. The dataset comprises labeled [examples](https://docs.smith.langchain.com/langsmith/evaluation-concepts#examples) containing questions and expected answers:

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

### Step 2: Define Your Application

Next, define the application you want to evaluate. Here's an example QA application:

```python
from openjudge.models.openai_chat_model import OpenAIChatModel
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

!!! info "Tracing Support"
    You can optionally enable tracing to capture the inputs and outputs of each step in the pipeline. Refer to the [LangSmith documentation](https://docs.smith.langchain.com/) for details.

### Step 3: Build Evaluators

Now wrap OpenJudge graders as LangSmith-compatible evaluators. First, select the appropriate grader based on your evaluation needs:

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

    The simplest approach is to create a wrapper function that converts OpenJudge graders to LangSmith evaluators:

    ```python
    from typing import Callable, Dict, Any, Union, Awaitable
    from openjudge.graders.base_grader import BaseGrader
    from openjudge.graders.schema import GraderResult, GraderScore, GraderRank, GraderError
    from openjudge.utils.mapping import parse_data_with_mapper

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
    ```

    Then use it with your graders:

    ```python
    import asyncio
    from openjudge.graders.common.relevance import RelevanceGrader
    from openjudge.graders.common.correctness import CorrectnessGrader
    from openjudge.models.openai_chat_model import OpenAIChatModel

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

=== "Batch Evaluation (Recommended)"

    For more complex situations involving multiple graders, create a batch evaluator class that uses OpenJudge's `GradingRunner`:

    ```python
    from openjudge.runner.grading_runner import GradingRunner
    from openjudge.graders.common.correctness import CorrectnessGrader
    from openjudge.graders.common.relevance import RelevanceGrader
    from openjudge.models.openai_chat_model import OpenAIChatModel
    from openjudge.graders.schema import GraderScore, GraderError
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
    ```

    Then use it:

    ```python
    from langsmith.evaluation import evaluate

    # Define comprehensive mapper for the batch evaluator
    # Maps LangSmith data format to OpenJudge format
    mapper = {
        "query": "inputs.question",
        "response": "outputs.answer",
        "reference_response": "reference_outputs.expected_answer"
    }

    # Create batch evaluator
    batch_evaluator = LangSmithBatchEvaluator(mapper=mapper)

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

### Step 4: View Results

Each invocation of [evaluate()](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate) creates an Experiment which can be viewed in the LangSmith UI or queried via the SDK. Evaluation scores are stored against each actual output as feedback.

You can access the results programmatically:

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

## Related Resources

- [OpenJudge Built-in Graders](../built_in_graders/overview.md) — Explore 50+ available graders for immediate use
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific evaluation logic
- [LangSmith Evaluation Guide](https://docs.langchain.com/langsmith/evaluate-llm-application) — Core evaluation concepts
