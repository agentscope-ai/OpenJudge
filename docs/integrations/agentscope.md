# AgentScope Integration

This guide walks you through building robust evaluation benchmarks for AgentScope using OpenJudge. By connecting these two tools, you can harness OpenJudge's 50+ built-in graders to deliver thorough automated quality assessment of your Multi-Agent systems directly within the AgentScope `MetricBase` architecture.


## Overview

While AgentScope provides a robust, event-driven architecture for *orchestrating* agents, the challenge of *validating* their performance often falls on ad-hoc scripts or simple string matching. OpenJudge fills this void by injecting a professional-grade, "Judge-as-a-Service" layer directly into your agent workflow.

Integrating OpenJudge fundamentally expands AgentScope's capabilities in three key dimensions:

* **From Ad-hoc to Standardized Assessment**: Instead of manually engineering prompts for every evaluation check (e.g., "Check if this answer is polite"), you gain immediate access to **50+ battle-tested Graders**.
* **Multi-Dimensional Insight**: AgentScope excels at managing agent interactions; OpenJudge excels at judging them. This integration allows you to evaluate not just the final output, but the **entire trajectory** of the agent's thought process, detecting subtle hallucinations, logical fallacies, or safety violations that simple metrics miss.
* **Enterprise-Ready Scalability**: By leveraging OpenJudge's optimized evaluation kernels, you transform AgentScope from a development framework into a **self-optimizing ecosystem**. You can run comprehensive benchmarks on large-scale datasets with the confidence that the "Judge" is as sophisticated as the "Agent" being tested.

The integration follows a simple three-step flow:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   AgentScope    │      │    OpenJudge    │      │   AgentScope    │
│ (Task/Solution) │─────▶│    (Wrapper)    │─────▶│ (MetricResult)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘

```

1. **Define Wrapper** — Create an adapter class that inherits from AgentScope's `MetricBase`.
2. **Configure Tasks** — Initialize tasks with OpenJudge graders and data mappers.
3. **Run Evaluation** — Execute the benchmark using AgentScope's `GeneralEvaluator`.

## Prerequisites

!!! warning "Required Configuration"
Make sure to set all required environment variables before running the code.

**Install dependencies:**

```bash
pip install py-openjudge agentscope

```

**Configure environment variables:**

```bash
# OpenAI API configuration (required for LLM-based graders)
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1" # Optional

```

## Integration Steps

### Step 1: Define the Metric Wrapper

To make OpenJudge compatible with AgentScope, we need a wrapper class. This class inherits from `MetricBase` and translates AgentScope's  `SolutionOutput` into the format OpenJudge expects.

```python
from agentscope.evaluate import MetricBase, MetricResult, MetricType, SolutionOutput
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore, GraderError
from openjudge.utils.mapping import parse_data_with_mapper

class OpenJudgeMetric(MetricBase):
    """
    A wrapper that converts an OpenJudge grader into an AgentScope Metric.
    """

    def __init__(
        self,
        grader_cls: type[BaseGrader],
        item: dict,
        mapper: dict | None = None,
        name: str | None = None,
        description: str | None = None,
        **grader_kwargs
    ):
        """
        Initialize the wrapper.

        Args:
            grader_cls: The OpenJudge grader class to instantiate
            item: The task item containing metadata/ground truth
            mapper: Dictionary mapping grader keys to task/solution paths
            **grader_kwargs: Arguments for the grader (e.g., model config)
        """
        # Instantiate the OpenJudge grader
        self.grader = grader_cls(**grader_kwargs)

        super().__init__(
            name=name or self.grader.name,
            metric_type=MetricType.NUMERICAL,
            description=description or self.grader.description
        )

        self.item = item
        self.mapper = mapper or {}

    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        """Execute the wrapped OpenJudge grader against the agent solution."""
        if not solution.success:
            return MetricResult(name=self.name, result=0.0, message="Solution failed")

        try:
            # Combine Task Context (item) and Agent Output (solution)
            combined_data = {
                "item": self.item,
                "solution": {
                    "output": solution.output,
                    "meta": solution.meta,
                    "trajectory": getattr(solution, 'trajectory', [])
                }
            }

            # Parse data using the mapper
            grader_inputs = parse_data_with_mapper(combined_data, self.mapper)

            # Run the OpenJudge grader
            result = await self.grader.aevaluate(**grader_inputs)

            # Convert to AgentScope MetricResult
            if isinstance(result, GraderScore):
                return MetricResult(
                    name=self.name,
                    result=result.score,
                    message=result.reason or ""
                )
            elif isinstance(result, GraderError):
                return MetricResult(name=self.name, result=0.0, message=f"Error: {result.error}")
            else:
                return MetricResult(name=self.name, result=0.0, message="Unknown result type")

        except Exception as e:
            return MetricResult(name=self.name, result=0.0, message=f"Exception: {str(e)}")

```

### Step 2: Configure Benchmark & Mappers

When creating your AgentScope `Task` objects, you must define a **Mapper**. This tells the wrapper how to extract the `query`, `response`, and `context` from your specific data structure.

!!! info "Mapper Logic"
The mapper uses dot-notation. `item` refers to your task data, and `solution` refers to the agent's output.

**Example: Building a QA Benchmark**

```python
from agentscope.evaluate import Task, BenchmarkBase
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.common.correctness import CorrectnessGrader
import os

# 1. Define your raw data
QA_DATA = [
    {
        "id": "qa_task_1",
        "input": "What are the health benefits of regular exercise?",
        "reference_output": "Regular exercise improves cardiovascular health, strengthens muscles and bones, helps maintain a healthy weight, and can improve mental health by reducing anxiety and depression.",
        "ground_truth": "Answers should cover physical and mental health benefits",
        "difficulty": "medium",
        "category": "health"
    },
]

class QABenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(name="QA Benchmark")
        self.dataset = self._load_data()

    def _load_data(self):
        dataset = []
        model_config = {
            "model": "qwen3-32b", # Or your preferred model
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL")
        }

        for item in QA_DATA:
            # 2. Define the Mapping
            # Maps OpenJudge standard fields -> AgentScope paths
            mapper={
                "query": "item.input",
                "response": "solution.output",
                "context": "item.ground_truth",
                "reference_response": "item.reference_output"
            },

            # 3. Instantiate Metrics
            relevance_metric = OpenJudgeMetric(
                grader_cls=RelevanceGrader,
                item=item,
                mapper=mapper,
                name="Relevance",
                model=model_config
            )

            correctness_metric = OpenJudgeMetric(
                grader_cls=CorrectnessGrader,
                item=item,
                mapper=mapper,
                name="Correctness",
                model=model_config
            )

            # 4. Create Task
            task = Task(
                id=item["id"],
                input=item["input"],
                ground_truth=item["ground_truth"],
                metrics=[relevance_metric, correctness_metric]
            )
            dataset.append(task)

        return dataset

```

### Step 3: Run Evaluation

Finally, use AgentScope's `GeneralEvaluator` to run your agent against the benchmark. The `OpenJudgeMetric` will automatically handle the scoring during the process.

```python
import asyncio
from agentscope.evaluate import GeneralEvaluator, FileEvaluatorStorage

# Your agent solution function
async def my_agent_solution(task, pre_hook=None):
    # ... Agent logic here ...
    return SolutionOutput(success=True, output="Agent generated response...")

async def run_eval():
    benchmark = QABenchmark()

    evaluator = GeneralEvaluator(
        name="OpenJudge Eval",
        benchmark=benchmark,
        storage=FileEvaluatorStorage(save_dir="./results"),
    )

    # Execute
    await evaluator.run(my_agent_solution)

if __name__ == "__main__":
    asyncio.run(run_eval())

```

### Step 4: View Results

After the evaluation completes, AgentScope saves the results to the directory specified in `FileEvaluatorStorage` (defaulting to `./results` in the example above). The results are saved as JSON files, which include the metric scores and the specific reasoning provided by the OpenJudge graders.

You can analyze these results by reading the JSON output directly or by loading them into a DataFrame for statistical analysis.

## Related Resources

* [OpenJudge Built-in Graders](../built_in_graders/overview.md) — Explore 50+ available graders for immediate use
* [AgentScope Evaluation Docs](https://doc.agentscope.io/tutorial/task_eval.html) — Learn more about the underlying AgentScope evaluation architectur
* [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific evaluation logic
