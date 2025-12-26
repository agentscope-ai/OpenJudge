# Evaluate AI Agents

Assess AI agent performance at three levels: **Final Response** (end results), **Single Step** (individual actions), and **Trajectory** (execution paths). This guide helps you identify failure points, optimize costs, and improve agent reliability.

!!! note "Additional Resources"
    For detailed grader documentation, see [Built-in Graders](../built_in_graders/overview.md).

## Why Evaluate AI Agents?

AI agents operate autonomously through complex reasoning loops, making multiple tool calls and decisions before reaching a final answer. This multi-step nature creates unique evaluation challenges—a wrong tool selection early on can cascade into complete task failure.

**Systematic evaluation enables you to:**

- **Identify failure points** — Pinpoint issues in planning, tool selection, or execution
- **Optimize costs** — Reduce unnecessary tool calls and LLM iterations
- **Ensure reliability** — Validate performance before deployment
- **Continuously improve** — Drive enhancements through data-driven insights

## Choose the Right Evaluation Granularity

| Granularity | What It Measures | When to Use |
|-------------|------------------|-------------|
| **Final Response** | Overall task success and answer quality | Production monitoring, A/B testing |
| **Single Step** | Individual action quality (tool calls, planning) | Debugging failures, prompt engineering |
| **Trajectory** | Multi-step reasoning paths and efficiency | Cost optimization, training reward models |

!!! tip "Evaluation Strategy"
    Start with **Final Response** evaluation to establish baseline success rates. When failures occur, use **Single Step** evaluation to pinpoint root causes. Use **Trajectory** evaluation to detect systemic issues like loops or inefficiencies.


## Evaluate Final Response

Assess the end result of agent execution to determine if the agent successfully completed the user's task.

### Step 1: Choose a Grader

Suppose you want to evaluate: **Is the agent's final answer factually correct compared to a reference answer?**

This is a **correctness** evaluation task. OpenJudge provides the `CorrectnessGrader` for exactly this purpose—it compares the response against a reference and scores accuracy on a 1-5 scale.

| Your Scenario | Recommended Grader |
|---------------|-------------------|
| Is the answer correct? | `CorrectnessGrader` |
| Does the response answer the question? | `RelevanceGrader` |
| Does the response contain hallucinations? | `HallucinationGrader` |
| Is the response harmful or unsafe? | `HarmfulnessGrader` |

For a complete list of available graders, see [Built-in Graders](../built_in_graders/overview.md).

In this example, we'll use `CorrectnessGrader` to evaluate the agent's final answer.

### Step 2: Initialize the Model

=== "Environment Variables"

    ```python
    from openjudge.models import OpenAIChatModel

    # Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
    model = OpenAIChatModel(model="qwen3-32b")
    ```

=== "Pass Directly"

    ```python
    from openjudge.models import OpenAIChatModel

    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    ```

### Step 3: Prepare Your Data

Prepare a dictionary with `query` and `response` fields:

```python
dataset = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris."
    }
]
```

### Step 4: Run Evaluation

```python
import asyncio
from openjudge.graders.common import CorrectnessGrader
from openjudge.models import OpenAIChatModel

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    grader = CorrectnessGrader(model=model)

    # Prepare data
    data = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris."
    }

    # Evaluate
    result = await grader.aevaluate(**data)
    print(result)

asyncio.run(main())
```

**Output:**

```python
GraderScore(
    name='correctness',
    score=5.0,
    reason="The response correctly states that the capital of France is Paris, which is factually consistent with the reference response 'Paris'. The added phrasing 'The capital of France is' provides appropriate context without contradicting or distorting the reference."
)
```


## Evaluate Single Step

Assess individual agent decisions in isolation—one tool call, one planning step, or one memory retrieval at a time.

### Step 1: Choose a Grader

Suppose you want to evaluate: **Did the agent select the most appropriate tool for the current sub-task?**

This is a **tool selection** evaluation task. OpenJudge provides the `ToolSelectionGrader` for exactly this purpose—it assesses whether the chosen tool matches the task requirements.

| Your Scenario | Recommended Grader |
|---------------|-------------------|
| Did the agent select the right tool? | `ToolSelectionGrader` |
| Did the tool call succeed? | `ToolCallSuccessGrader` |
| Is the plan feasible? | `PlanFeasibilityGrader` |
| Is the memory retrieval accurate? | `MemoryAccuracyGrader` |
| Is the reflection accurate? | `ReflectionAccuracyGrader` |

For a complete list of available graders, see [Agent Graders](../built_in_graders/agent_graders.md).

In this example, we'll use `ToolSelectionGrader` to evaluate the agent's tool choice.

### Step 2: Initialize the Model

=== "Environment Variables"

    ```python
    from openjudge.models import OpenAIChatModel

    # Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
    model = OpenAIChatModel(model="qwen3-32b")
    ```

=== "Pass Directly"

    ```python
    from openjudge.models import OpenAIChatModel

    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    ```

### Step 3: Prepare Your Data

Single Step graders require specific fields extracted from your agent traces. Prepare a dictionary with `query`, `tool_definitions`, and `tool_calls`:

```python
data = {
    "query": "What's 15% tip on a $45 bill?",
    "tool_definitions": [
        {"name": "calculator", "description": "Perform mathematical calculations"},
        {"name": "search_web", "description": "Search the web for information"}
    ],
    "tool_calls": [
        {"name": "calculator", "arguments": '{"expression": "45 * 0.15"}'}
    ]
}
```

!!! tip "Extracting from Agent Traces"
    If your data is in OpenAI messages format, you'll need to extract the relevant fields. See the complete example below for a mapper function.

### Step 4: Run Evaluation

```python
import asyncio
from openjudge.graders.agent import ToolSelectionGrader
from openjudge.models import OpenAIChatModel

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    grader = ToolSelectionGrader(model=model)

    # Prepare data
    data = {
        "query": "What's 15% tip on a $45 bill?",
        "tool_definitions": [
            {"name": "calculator", "description": "Perform mathematical calculations"},
            {"name": "search_web", "description": "Search the web for information"}
        ],
        "tool_calls": [
            {"name": "calculator", "arguments": '{"expression": "45 * 0.15"}'}
        ]
    }

    # Evaluate
    result = await grader.aevaluate(**data)
    print(result)

asyncio.run(main())
```

**Output:**

```python
GraderScore(
    name='tool_selection',
    score=5.0,
    reason="The agent selected the 'calculator' tool with the expression '45 * 0.15', which is the most direct and efficient tool for computing a percentage-based tip. The query is purely mathematical, requiring no external information. The calculator tool is fully capable of performing this arithmetic operation accurately."
)
```


## Evaluate Trajectory

Assess the entire sequence of agent actions to determine if the agent took an optimal path without loops or redundant steps.

### Step 1: Choose a Grader

Suppose you want to evaluate: **Did the agent complete the task efficiently without unnecessary steps or loops?**

This is a **trajectory** evaluation task. OpenJudge provides the `TrajectoryComprehensiveGrader` for exactly this purpose—it analyzes the full execution path for efficiency and correctness.

For a complete list of available graders, see [Agent Graders](../built_in_graders/agent_graders.md).

In this example, we'll use `TrajectoryComprehensiveGrader` to evaluate the agent's execution path.

### Step 2: Initialize the Model

=== "Environment Variables"

    ```python
    from openjudge.models import OpenAIChatModel

    # Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
    model = OpenAIChatModel(model="qwen3-32b")
    ```

=== "Pass Directly"

    ```python
    from openjudge.models import OpenAIChatModel

    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    ```

### Step 3: Prepare Your Data

Prepare a full agent trajectory in OpenAI messages format:

```python
data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": '{"temp": 22, "condition": "sunny"}'
        },
        {
            "role": "assistant",
            "content": "The weather in Tokyo is sunny with 22°C."
        }
    ]
}
```

### Step 4: Run Evaluation

```python
import asyncio
from openjudge.graders.agent import TrajectoryComprehensiveGrader
from openjudge.models import OpenAIChatModel

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    grader = TrajectoryComprehensiveGrader(model=model)

    # Prepare data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": '{"temp": 22, "condition": "sunny"}'
            },
            {
                "role": "assistant",
                "content": "The weather in Tokyo is sunny with 22°C."
            }
        ]
    }

    # Evaluate
    result = await grader.aevaluate(**data)
    print(result)

asyncio.run(main())
```

**Output:**

```python
GraderScore(
    name='trajectory_comprehensive',
    score=1.0,
    reason="The agent efficiently completed the task in a single tool call. It correctly identified the need for weather information, selected the appropriate tool, and provided a clear, accurate response based on the tool output. No unnecessary steps or loops were detected."
)
```


## Batch Evaluation with GradingRunner

For evaluating multiple agent traces efficiently, use `GradingRunner` to run graders concurrently with automatic progress tracking:

```python
import asyncio
from openjudge.graders.agent import ToolSelectionGrader
from openjudge.models import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolSelectionGrader(model=model)

    # Define mapper to extract grader inputs from agent traces
    def extract_tool_inputs(data: dict) -> dict:
        messages = data["messages"]
        query = next((m["content"] for m in messages if m["role"] == "user"), "")
        tool_calls = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls.append({
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    })
        return {
            "query": query,
            "tool_definitions": data["available_tools"],
            "tool_calls": tool_calls
        }

    # Configure runner with mapper
    runner = GradingRunner(
        grader_configs={
            "tool_selection": GraderConfig(
                grader=grader,
                mapper=extract_tool_inputs
            )
        },
        max_concurrency=16,
        show_progress=True
    )

    # Prepare dataset (agent traces)
    dataset = [
        {   # Bad case: should use calculator, not search_web
            "messages": [
                {"role": "user", "content": "What's 15% tip on a $45 bill?"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "search_web", "arguments": '{"query": "15% tip on $45"}'}}]}
            ],
            "available_tools": [
                {"name": "calculator", "description": "Perform mathematical calculations"},
                {"name": "search_web", "description": "Search the web for information"}
            ]
        },
        {   # Good case: correctly uses get_weather
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}]}
            ],
            "available_tools": [
                {"name": "get_weather", "description": "Get weather information"},
                {"name": "search_web", "description": "Search the web for information"}
            ]
        },
    ]

    # Run batch evaluation
    results = await runner.arun(dataset)

    # Print results
    for i, result in enumerate(results["tool_selection"]):
        print(f"Trace {i}: Score={result.score}")

asyncio.run(main())
```

**Output:**

```
Trace 0: Score=2.0   # Wrong tool: used search_web instead of calculator
Trace 1: Score=5.0   # Correct tool: used get_weather for weather query
```

For more details on batch evaluation, data mapping, and result aggregation, see [Run Grading Tasks](../running_graders/run_tasks.md).


## Next Steps

- [Built-in Graders](../built_in_graders/overview.md) — Detailed documentation for all available graders
- [Agent Graders](../built_in_graders/agent_graders.md) — Learn about the built-in agent graders
- [Run Grading Tasks](../running_graders/run_tasks.md) — Batch evaluation with concurrency and progress tracking