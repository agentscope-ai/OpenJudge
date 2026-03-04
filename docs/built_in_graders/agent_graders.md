# Agent Graders

Evaluate AI agent behavior across actions, tools, memory, planning, reflection, and trajectories. These graders help you assess decision quality, detect failures, and optimize agent performance at every step.

## Overview

| Category | Grader | Purpose | Type | Score Range | Key Use Case |
|----------|--------|---------|------|-------------|--------------|
| **Action** | `ActionAlignmentGrader` | Evaluates action-plan consistency | LLM-Based | {0, 1} | ReAct agents, step-by-step reasoning |
| | `ActionLoopDetectionGrader` | Detects repetitive actions | Code-Based | [0, 1] | Multi-step exploration tasks |
| **Tool** | `ToolSelectionGrader` | Assesses tool choice quality | LLM-Based | 1-5 | Function calling agents |
| | `ToolCallAccuracyGrader` | Evaluates tool call accuracy | LLM-Based | 1-5 | API-based assistants |
| | `ToolCallSuccessGrader` | Checks technical execution success | LLM-Based | {0, 1} | Production agent monitoring |
| | `ToolParameterCheckGrader` | Validates parameter correctness | LLM-Based | {0, 1} | Slot-filling dialogues |
| | `ToolCallStepSequenceMatchGrader` | Multi-step tool sequence matching with step alignment | Code-Based | [0, 1] | Complex multi-turn agent benchmarks |
| | `ToolCallPrecisionRecallMatchGrader` | Simple precision/recall for flat tool call lists | Code-Based | [0, 1] | Single-step tool call evaluation |
| **Memory** | `MemoryAccuracyGrader` | Validates memory factuality | LLM-Based | {0, 1} | Memory-augmented agents |
| | `MemoryDetailPreservationGrader` | Checks detail retention | LLM-Based | {0, 1} | Long-horizon tasks |
| | `MemoryRetrievalEffectivenessGrader` | Assesses memory retrieval | LLM-Based | {0, 1} | RAG-based agents |
| **Plan** | `PlanFeasibilityGrader` | Evaluates plan feasibility | LLM-Based | {0, 1} | Task planning agents |
| **Reflection** | `ReflectionAccuracyGrader` | Validates reflection accuracy | LLM-Based | {0, 1} | Self-correcting agents |
| | `ReflectionOutcomeUnderstandingGrader` | Checks outcome understanding | LLM-Based | {0, 1} | Error recovery scenarios |
| | `ReflectionProgressAwarenessGrader` | Assesses progress awareness | LLM-Based | {0, 1} | Goal-tracking agents |
| **Observation** | `ObservationInformationGainGrader` | Measures information gain | Code-Based | [0, 1] | Exploration efficiency |
| **Trajectory** | `TrajectoryAccuracyGrader` | Evaluates trajectory accuracy in achieving goals | LLM-Based | 1-3 | Task completion assessment |
| | `TrajectoryComprehensiveGrader` | Comprehensive trajectory evaluation | LLM-Based | [0, 1] | End-to-end agent testing |

## Performance

Benchmark results using qwen3-max on agent evaluation tasks:

| Category | Grader | Samples | Preference Accuracy | Source |
|----------|--------|---------|---------------------|--------|
| Action | ActionAlignmentGrader | 8 | 88% | ALFWorld, WebShop, GAIA |
| Tool | ToolCallAccuracyGrader | 40 | 90% | API-Bank |
| Tool | ToolCallSuccessGrader | 20 | 95% | API-Bank |
| Tool | ToolParameterCheckGrader | 20 | 75% | API-Bank |
| Tool | ToolSelectionGrader | 20 | 85% | API-Bank |
| Memory | MemoryAccuracyGrader | 18 | 78% | ALFWorld, WebShop, GAIA |
| Memory | MemoryDetailPreservationGrader | 25 | 76% | ALFWorld, WebShop, GAIA |
| Memory | MemoryRetrievalEffectivenessGrader | 4 | 100% | ALFWorld |
| Plan | PlanFeasibilityGrader | 7 | 86% | ALFWorld, GAIA |
| Reflection | ReflectionAccuracyGrader | 2 | 100% | ALFWorld |
| Reflection | ReflectionOutcomeUnderstandingGrader | 23 | 78% | ALFWorld, GAIA |
| Reflection | ReflectionProgressAwarenessGrader | 28 | 74% | ALFWorld, WebShop, GAIA |

!!! note "Performance Metrics"
    Preference Accuracy measures alignment with human-annotated preference labels (positive and negative samples) on agent evaluation tasks. Higher is better.

**Benchmark Sources:**

- **API-Bank**: [A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244) (EMNLP 2023) — A benchmark with 73 API tools and 314 tool-use dialogues for evaluating LLM tool utilization capabilities.
- **ALFWorld, WebShop, GAIA**: Evaluation datasets from [Where LLM Agents Fail and How They can Learn From Failures](https://arxiv.org/abs/2509.25370) — A systematic study of agent failure modes with the AgentErrorTaxonomy covering memory, reflection, planning, and action modules.


## Action Graders

### ActionAlignmentGrader

Evaluates whether agent actions align with stated plans or reasoning.

**Use this grader to:**

1. Verify consistency between planning and execution
2. Debug agent decision-making processes
3. Ensure actions follow stated intentions

**Evaluation criteria:** Direct plan implementation, correct object targeting, goal contribution, logical sequence, and constraint respect.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning statement |
| `action` | str | Yes | Agent's executed action |
| `history` | List[dict] | No | Previous step dictionaries for context |
| `context` | str | No | Task context (description, environment, available actions) |

**Scoring:**
- `1.0`: Good alignment - action follows plan logically
- `0.0`: Poor alignment - action inconsistent with plan

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ActionAlignmentGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ActionAlignmentGrader(model=model)

    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="open drawer 1",
        context="Task: Find the key to unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - good alignment
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The action 'open drawer 1' directly implements the stated plan 'I will open drawer 1 to find the key.' It targets the correct object (drawer 1), contributes to achieving the goal of finding the key, follows the logical order outlined in the plan, and respects any implied preconditions (e.g., needing to open the drawer to access its contents). The alignment is clear and direct, so confidence is high.
```

### ActionLoopDetectionGrader

Detects repetitive or similar actions in agent sequences.

**Use this grader to:**

1. Identify when agents get stuck in loops
2. Detect inefficient exploration strategies
3. Debug stuck agents in multi-step tasks

**Evaluation criteria:** Compares all pairs of action signatures for similarity.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Message list containing agent interactions |
| `similarity_threshold` | float | No | Threshold to consider actions similar (default: 1.0) |

**Scoring:**
- `1.0`: No loops detected
- `0.0`: Many similar action pairs found
- Score computed as: `1.0 - (similar_pairs / total_pairs)`

**Example:**

```python
import asyncio
from openjudge.graders.agent import ActionLoopDetectionGrader

async def main():
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "search", "arguments": '{"query": "python"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Results..."},
        {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "search", "arguments": '{"query": "python"}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "Results..."},
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Score: {result.score}")   # Lower score indicates loop
    print(f"Similar pairs: {result.metadata['similar_pair_count']}")

asyncio.run(main())
```

**Output:**

```
Score: 0.0
Similar pairs: 1
```


## Tool Graders

### ToolSelectionGrader

Evaluates tool selection quality for addressing user queries.

**Use this grader to:**

1. Assess tool choice appropriateness
2. Evaluate agent decision-making quality
3. Compare different agent architectures

**Evaluation criteria:** Tool relevance, selection completeness, efficiency, and understanding of tool capabilities.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | User query or conversation history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Available tool definitions |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tools actually selected by agent |

**Scoring:**
- `5`: Optimal tool selection - most direct and efficient
- `4`: Reasonable selection - can complete task but not optimal
- `3`: Acceptable - related but not direct match
- `2`: Poor - clearly mismatched with task
- `1`: Incorrect - no tool selected or completely irrelevant

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ToolSelectionGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolSelectionGrader(model=model)

    result = await grader.aevaluate(
        query="Find all Python files modified in the last week",
        tool_definitions=[
            {"name": "search_files", "description": "Search for files by pattern"},
            {"name": "git_log", "description": "Get git commit history"},
            {"name": "read_file", "description": "Read file contents"}
        ],
        tool_calls=[
            {"name": "search_files", "arguments": {"pattern": "*.py"}},
            {"name": "git_log", "arguments": {"days": 7}}
        ]
    )

    print(f"Score: {result.score}")   # 4-5 - good tool selection
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The selected tools are highly relevant and directly address the user's query. The 'search_files' tool with the pattern '*.py' effectively identifies all Python files in the system, while the 'git_log' tool with the argument 'days: 7' retrieves the commit history for the last week, which can be used to determine which of those Python files were modified recently. Together, these tools provide a complete and efficient solution without including any unnecessary or redundant tools. The selection demonstrates a clear understanding of both the task intent and the capabilities of the available tools.
```

### ToolCallAccuracyGrader

Evaluates tool call accuracy including parameter correctness and query relevance.

**Use this grader to:**

1. Validate tool call correctness
2. Assess parameter extraction accuracy
3. Evaluate agent tool-use capability

**Evaluation criteria:** Tool relevance to query and parameter correctness according to definitions.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | Query or chat history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions with parameters |
| `tool_calls` | List[Dict[str, Any]] | No | Tool calls to evaluate (or provide `response`) |
| `response` | str or List[Dict] | No | Response containing tool calls |

**Scoring:**
- `5`: Fully relevant, all parameters correct
- `4`: Relevant, tools returned errors but agent retried successfully
- `3`: Relevant but unnecessary/excessive calls
- `2`: Partially relevant, insufficient tools or incorrect parameters
- `1`: Irrelevant or tool names not found in definitions

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ToolCallAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolCallAccuracyGrader(model=model)

    conversation = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]

    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {"location": "City name"}
        }
    ]

    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls
    )

    print(f"Score: {result.score}")   # 5.0 - accurate tool call
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The tool call 'get_weather' is fully relevant to the user's query about the weather in New York. The name of the tool call matches one of the function names in the tool definitions, and the parameter 'location' with the value 'New York' is correctly extracted from the conversation and aligns with the description in the tool definition.
```

### ToolCallSuccessGrader

Evaluates technical execution success of tool calls (no errors, exceptions, or timeouts).

**Use this grader to:**

1. Detect technical failures in tool execution
2. Monitor agent reliability
3. Debug tool integration issues

**Evaluation criteria:** Checks for technical execution success (no errors, exceptions, or timeouts). Does not evaluate business correctness.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions for context |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tool calls to evaluate (name and arguments) |
| `tool_responses` | str or List[str] | Yes | Tool responses corresponding to each tool call |

**Scoring:**
- `1.0`: All tool calls successful
- `0.0`: At least one tool call failed

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ToolCallSuccessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolCallSuccessGrader(model=model)

    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {"location": "City name"}
        }
    ]

    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
    ]

    tool_responses = [
        "The weather in New York is sunny and 25 degrees Celsius."
    ]

    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
        tool_responses=tool_responses
    )

    print(f"Score: {result.score}")   # 1.0 - successful
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The tool call executed successfully, returned a non-empty result, and did not contain any error messages or exceptions.
```

### ToolParameterCheckGrader

Evaluates parameter extraction accuracy from user queries.

**Use this grader to:**

1. Validate parameter extraction accuracy
2. Ensure grounded parameter values
3. Detect hallucinated parameters

**Evaluation criteria:** Parameter completeness, accuracy, grounding, and correct mapping.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | User query or conversation history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions with parameter specifications |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tool calls made by the agent |

**Scoring:**
- `1.0`: All parameters correct and complete
- `0.0`: Parameters have issues (missing, incorrect, or fabricated)

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ToolParameterCheckGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolParameterCheckGrader(model=model)

    result = await grader.aevaluate(
        query="Search for Python files in the src directory",
        tool_definitions=[
            {
                "name": "search_files",
                "parameters": {"pattern": "str", "directory": "str"}
            }
        ],
        tool_calls=[
            {
                "name": "search_files",
                "arguments": {"pattern": "*.py", "directory": "src"}
            }
        ]
    )

    print(f"Score: {result.score}")   # 1.0 - correct parameters
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The tool call correctly extracted all required parameters from the user query. The 'pattern' parameter was set to '*.py', which accurately reflects the intent to search for Python files. The 'directory' parameter was set to 'src', matching the specified directory in the query. Both parameters are present, grounded in the query, and formatted correctly as strings. There are no hallucinations or missing parameters, and the data types align with the tool's definition. The tool call is fully executable with correct parameters.
```

### ToolCallStepSequenceMatchGrader

Evaluates multi-step tool call sequences with step-by-step alignment against reference sequences. Designed for complex multi-turn agent scenarios where tool calls are organized by steps (turns).

**Use this grader for:**

- **Multi-step agent benchmarks** — Evaluate agents that make multiple tool calls across different turns
- **Step-aligned trajectory comparison** — Compare tool sequences where order and grouping by steps matter
- **Complex agentic workflows** — Validate multi-turn conversations with tool calls organized by assistant turns

**Evaluation criteria:** Supports step-by-step matching or Jaccard similarity. Strict mode matches name + parameters; loose mode matches name only.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Agent's message history with tool calls |
| `reference_tool_calls` | List[List[Dict[str, Any]]] | Yes | Ground truth reference tool sequence by steps |
| `strict_mode` | bool | No | Match name + parameters (True) or name only (False), default: True |
| `use_jaccard_similarity` | bool | No | Use Jaccard similarity ignoring order (True) or step-by-step (False), default: True |

**Scoring:**
- **Strict mode with Jaccard**: Intersection over union of (tool_name, parameters) pairs
- **Loose mode with Jaccard**: Intersection over union of tool names
- **Step-by-step mode**: Average F1 score across steps
- Range: 0.0 (no match) to 1.0 (perfect match)

**Example:**

```python
import asyncio
from openjudge.graders.agent import ToolCallStepSequenceMatchGrader

async def main():
    grader = ToolCallStepSequenceMatchGrader(
        strict_mode=True,
        use_jaccard_similarity=True
    )

    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "search", "arguments": '{"query": "python"}'}}
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "Results..."},
    ]

    reference_tool_calls = [
        [
            {"name": "search", "arguments": {"query": "python"}}
        ]
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls
    )

    print(f"Score: {result.score}")   # 1.0 - perfect match
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: Tool call sequence evaluation (strict mode, jaccard): jaccard_similarity=1.000
```

### ToolCallPrecisionRecallMatchGrader

Computes precision or recall metrics for tool calls against reference.

**Use this grader for:**

- Simple tool call evaluation without step/order consideration
- Computing precision (correctness of predictions) or recall (coverage of reference)
- Flexible matching with or without argument comparison

**Evaluation criteria:** Compares predicted tool calls against reference tool calls using set-based matching.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_calls` | List[Dict[str, Any]] | Yes | Predicted tool calls to evaluate |
| `reference_tool_calls` | List[Dict[str, Any]] | Yes | Ground truth reference tool calls |
| `metric_type` | str | No | "precision" or "recall" (default: "recall") |
| `match_arguments` | bool | No | Match name + arguments (True) or name only (False), default: False |

**Scoring:**
- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total references
- Range: 0.0 (no match) to 1.0 (perfect match)

**Example:**

```python
import asyncio
from openjudge.graders.agent import ToolCallPrecisionRecallMatchGrader

async def main():
    # Compute recall with loose matching (name only)
    grader = ToolCallPrecisionRecallMatchGrader(
        metric_type="recall",
        match_arguments=False
    )

    tool_calls = [
        {"name": "search", "arguments": {"query": "python"}},
        {"name": "calculate", "arguments": {"expr": "1+1"}}
    ]

    reference_tool_calls = [
        {"name": "search", "arguments": {"query": "test"}},
        {"name": "calculate", "arguments": {"expr": "2+2"}},
        {"name": "send_email", "arguments": {"to": "user@example.com"}}
    ]

    result = await grader.aevaluate(
        tool_calls=tool_calls,
        reference_tool_calls=reference_tool_calls
    )

    print(f"Score: {result.score}")   # 0.667 - 2 out of 3 references matched
    print(f"Precision: {result.metadata['precision']}")
    print(f"Recall: {result.metadata['recall']}")

asyncio.run(main())
```

**Output:**

```
Score: 0.6666666666666666
Precision: 1.0
Recall: 0.6666666666666666
```

**Strict matching example:**

```python
# Compute precision with strict matching (name + arguments)
grader = ToolCallPrecisionRecallMatchGrader(
    metric_type="precision",
    match_arguments=True
)

result = await grader.aevaluate(
    tool_calls=[{"name": "search", "arguments": {"query": "python"}}],
    reference_tool_calls=[{"name": "search", "arguments": {"query": "python"}}]
)

print(f"Score: {result.score}")   # 1.0 - exact match
```


## Memory Graders

### MemoryAccuracyGrader

Evaluates accuracy and factuality of stored memory content.

**Use this grader to:**

1. Validate memory system correctness
2. Ensure grounded information storage
3. Debug hallucination in memory

**Evaluation criteria:** Memory reflects actual observations, stores only factual details, and maintains accurate associations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous step dictionaries |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Accurate and factual memory
- `0.0`: Inaccurate or fabricated memory

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import MemoryAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet with three drawers.",
        memory="The cabinet is closed and has three drawers.",
        context="Task: Inventory room objects"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The memory accurately reflects the observation by recording only factual details present in the input. The agent correctly notes that the cabinet is 'closed' and has 'three drawers,' which are both explicitly mentioned in the observation. There are no interpretations, assumptions, or fabrications included in the memory. The information is consistent with what was observed, and all recorded elements are grounded in the provided context. This demonstrates good accuracy as per the rubrics.
```

### MemoryDetailPreservationGrader

Evaluates preservation of important details in stored memory.

**Use this grader to:**

1. Validate detail retention
2. Ensure actionable memory content
3. Debug information loss

**Evaluation criteria:** Storage of specific details, exact locations, numerical values, and important constraints.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous step dictionaries |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Important details preserved
- `0.0`: Important details lost or generalized

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import MemoryDetailPreservationGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryDetailPreservationGrader(model=model)

    result = await grader.aevaluate(
        observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
        memory="Cabinet 1 at (3.5, 2.1) has 5 red apples.",
        context="Task: Inventory items with precise locations"
    )

    print(f"Score: {result.score}")   # 1.0 - details preserved
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The agent successfully preserves all important details from the observation in its memory. It retains the specific location of Cabinet 1 with exact coordinates (3.5, 2.1), the quantity of items (5 apples), and the attribute (red). These details align directly with the rubrics for preserving spatial information, numerical values, and specific attributes. The memory is sufficiently detailed and actionable for future inventory-related tasks. Confidence is high because the preservation is explicit and matches the original observation precisely.
```

### MemoryRetrievalEffectivenessGrader

Evaluates effectiveness of memory retrieval for planning and decision-making.

**Use this grader to:**

1. Assess memory system effectiveness
2. Detect failure to use available information
3. Identify repetitive behavior due to poor retrieval

**Evaluation criteria:** Memory retrieval relevance, usage in planning, and avoidance of redundant exploration.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning |
| `observation` | str | Yes | Current environment observation |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Effective memory retrieval
- `0.0`: Ineffective retrieval or failure to use memory

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import MemoryRetrievalEffectivenessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    result = await grader.aevaluate(
        plan="I will use the key from drawer 1 to unlock the door.",
        observation="You are standing in the room with a locked door.",
        memory="The key was found in drawer 1 in step 3.",
        context="Task: Unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - effective retrieval
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The agent's plan effectively retrieves relevant information from memory by referencing the key found in drawer 1 during step 3. This demonstrates that the agent is using previously stored and correct information to inform its current action of unlocking the door. The plan aligns with the memory content, avoids repetition of past actions (no indication of trying other drawers), and is consistent with the observation of a locked door. The retrieval is current and accurate, showing strong memory effectiveness. Confidence is high because the connection between memory and plan is clear and directly supports the task at hand.
```


## Plan Graders

### PlanFeasibilityGrader

Evaluates logical soundness and feasibility of agent plans.

**Use this grader to:**

1. Validate agent planning capability
2. Ensure logical action sequences
3. Debug infeasible plans

**Evaluation criteria:** Causal logic, action order feasibility, executability, and prerequisite awareness.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning |
| `observation` | str | Yes | Current environment observation |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Feasible and logically sound
- `0.0`: Infeasible or illogical

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import PlanFeasibilityGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = PlanFeasibilityGrader(model=model)

    result = await grader.aevaluate(
        plan="I will first open the drawer to get the key, then use it to unlock the door.",
        observation="The drawer is closed. You don't have any items.",
        memory="The key is inside the drawer.",
        context="Task: Unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - feasible
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The plan is logically sound and feasible. It respects causal logic by first retrieving the key (which is inside the drawer) before attempting to unlock the door. The sequence of actions—opening the drawer, obtaining the key, and then unlocking the door—is in a correct and necessary order. The plan also accounts for the current environment state: the drawer is closed, and the agent does not yet have the key. Therefore, opening the drawer is a valid prerequisite action. The steps are consistent with the goal of unlocking the door and are executable given the described scenario. Confidence is high because all rubrics for feasibility are clearly satisfied.
```


## Reflection Graders

### ReflectionAccuracyGrader

Evaluates accuracy of agent reflections based on actual observations.

**Use this grader to:**

1. Validate agent self-assessment accuracy
2. Ensure grounded reflections
3. Debug hallucination in reasoning

**Evaluation criteria:** Reflections only mention observed objects, states, and details.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Accurate and grounded reflection
- `0.0`: Contains fabrications

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ReflectionAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        reflection="I observed a closed cabinet.",
        context="Task: Find objects in the room"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The reflection accurately summarizes the observation without adding, omitting, or fabricating any information. The agent mentions only what was observed: a closed cabinet. It does not introduce any additional objects, states, or details that were not present in the original observation. This demonstrates full compliance with all rubrics for reflection accuracy. Confidence is high because the reflection is directly and explicitly grounded in the observation.
```

### ReflectionOutcomeUnderstandingGrader

Evaluates correctness of action outcome interpretation in reflections.

**Use this grader to:**

1. Validate outcome interpretation accuracy
2. Detect fabricated or distorted understanding
3. Ensure evidence-based reasoning

**Evaluation criteria:** Strict factual accuracy checking of outcome interpretation.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Correct understanding - reflection accurately mirrors observation
- `0.0`: Poor understanding - factual distortion, failure misinterpretation, premature conclusions, scope overreach, inference leaps, fabrication, or format misinterpretation

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ReflectionOutcomeUnderstandingGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    result = await grader.aevaluate(
        observation="The drawer is now open. You see a key inside.",
        reflection="I successfully opened the drawer and found a key inside.",
        context="Task: Find the key"
    )

    print(f"Score: {result.score}")   # 1.0 - correct understanding
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The reflection accurately mirrors the observation: 'The drawer is now open. You see a key inside.' The agent correctly interprets this as a successful action (opening the drawer) and identifies the presence of the key, which aligns with the task objective of finding the key. There is no factual distortion, no unsupported inference, and no overreach in interpreting partial information. The agent does not claim to have seen all contents or make premature conclusions about absence. The reasoning is directly supported by the observation and demonstrates good understanding of both the outcome and its implications.
```

### ReflectionProgressAwarenessGrader

Evaluates accuracy of task progress awareness and sub-goal recognition.

**Use this grader to:**

1. Assess task progress tracking
2. Detect loop/stuck situations
3. Validate sub-goal awareness

**Evaluation criteria:** Correct identification of accomplishments, accurate distance-to-goal assessment, and recognition of all sub-goals.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context (critical for sub-goal tracking) |

**Scoring:**
- `1.0`: Accurate awareness - correctly identifies accomplishments, assesses distance to goal, recognizes all sub-goals
- `0.0`: Inaccurate awareness - misjudges progress, overlooks sub-goals, claims "almost done" while major requirements unmet

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ReflectionProgressAwarenessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionProgressAwarenessGrader(model=model)

    result = await grader.aevaluate(
        observation="You have collected 3 out of 5 required items.",
        reflection="Good progress! I'm about halfway through the task. Still need to find 2 more items.",
        context="Task: Collect 5 specific items from different locations"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate awareness
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The agent demonstrates accurate progress awareness by correctly identifying that it has collected 3 out of the 5 required items and acknowledging that 2 more are still needed. The reflection states, 'I'm about halfway through the task,' which is a realistic estimation given the current state. The agent does not overestimate its progress or ignore any critical sub-goals. It also shows awareness of the exact number of remaining tasks without substituting or omitting any specific item from the original task description. The reflection is concise but contains all necessary information to assess forward progress accurately. Confidence in this evaluation is high because the agent's self-assessment aligns with the observable facts and task constraints.
```


## Observation Graders

### ObservationInformationGainGrader

Measures information gain and redundancy in observation sequences.

**Use this grader to:**

1. Evaluate exploration efficiency
2. Detect redundant information gathering
3. Assess agent curiosity/exploration strategy

**Evaluation criteria:** Rewards novel observations, penalizes redundant ones based on similarity threshold.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Message list containing agent interactions |
| `similarity_threshold` | float | No | Redundancy threshold (default: 0.5) |

**Scoring:**
- `1.0`: High information gain, low redundancy
- `0.0`: High redundancy, low information gain
- Score based on average per-observation novelty with exponential penalty for similarity

**Example:**

```python
import asyncio
from openjudge.graders.agent import ObservationInformationGainGrader

async def main():
    grader = ObservationInformationGainGrader(similarity_threshold=0.5)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "look", "arguments": '{}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "You see a red box."},
        {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "look", "arguments": '{}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "You see a blue sphere."},
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Score: {result.score}")   # Higher = more novel observations
    print(f"Each turn similarity: {result.metadata['each_turn_similarity']}")

asyncio.run(main())
```

**Output:**

```
Score: 0.7857142857142857
Each turn similarity: [0.0, 0.42857142857142855]
```


## Trajectory Graders

### TrajectoryAccuracyGrader

Evaluates the accuracy of agent trajectories in solving user queries.

**Use this grader for:**

- Task completion assessment
- Trajectory quality evaluation
- Agent performance benchmarking

**Evaluation criteria:** Logical coherence between steps, goal progression, and efficiency.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | List of messages in OpenAI format representing the conversation trajectory |

**Scoring:**
- `3`: Successfully achieves the task goal without any steps unrelated to the task (reasonable extensions acceptable)
- `2`: Successfully achieves the task goal, but contains obvious unnecessary steps unrelated to the task
- `1`: Fails to achieve the task goal

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import TrajectoryAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = TrajectoryAccuracyGrader(model=model)

    messages = [
        {"role": "user", "content": "What's the weather like in New York?"},
        {"role": "assistant", "content": "I'll check the weather for you.",
         "tool_calls": [{"id": "1", "function": {"name": "get_weather", "arguments": '{"location": "New York"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Sunny, 72°F"},
        {"role": "assistant", "content": "The weather in New York is sunny, 72°F."}
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Score: {result.score}")   # 3.0 - goal achieved efficiently
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 3.0
Reason: The trajectory successfully achieves the task goal of providing weather information for New York. The agent made a single, relevant tool call to get_weather with the correct location parameter, received the weather data, and provided a clear response to the user. There are no unnecessary steps or actions unrelated to the task.
```

---

### TrajectoryComprehensiveGrader

Comprehensive evaluation of complete agent trajectories.

**Use this grader for:**

- End-to-end agent evaluation
- Holistic trajectory assessment
- Agent benchmark evaluation
- Production quality monitoring

**Evaluation criteria:** Step contribution, relevance, accuracy, and efficiency across the complete trajectory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Complete message history including system, user, assistant, tool |
| `resolution_threshold` | float | No | Threshold for success determination (default: 0.8) |

**Scoring:**

Each dimension uses 1-5 scale in prompts, normalized to 0-1:
- `5` → `1.0`: Excellent
- `4` → `0.75`: Good
- `3` → `0.5`: Acceptable
- `2` → `0.25`: Poor
- `1` → `0.0`: Very poor

Overall score is the average across all steps and dimensions.

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import TrajectoryComprehensiveGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = TrajectoryComprehensiveGrader(
        model=model,
        resolution_threshold=0.75
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Find Python files modified today"},
        {"role": "assistant", "content": "I'll search for Python files.",
         "tool_calls": [{"id": "1", "function": {"name": "search_files", "arguments": '{"pattern": "*.py"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Found: main.py, utils.py"},
        {"role": "assistant", "content": "I'll check their modification dates.",
         "tool_calls": [{"id": "2", "function": {"name": "get_file_info", "arguments": '{"files": ["main.py", "utils.py"]}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "main.py: today, utils.py: yesterday"},
        {"role": "assistant", "content": "Found 1 file modified today: main.py"}
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Overall Score: {result.score}")   # 0.0-1.0
    print(f"Is Resolved: {result.metadata['is_resolved']}")
    print(f"Avg Contribution: {result.metadata['avg_contribution']}")
    print(f"Avg Relevance: {result.metadata['avg_relevance']}")
    print(f"Avg Accuracy: {result.metadata['avg_accuracy']}")
    print(f"Avg Efficiency: {result.metadata['avg_efficiency']}")

    # Per-step details
    for step in result.metadata['step_evaluations']:
        print(f"Step {step['step_index']}: {step['step_reason']}")

asyncio.run(main())
```

**Output:**

```
Overall Score: 1.0
Is Resolved: True
Avg Contribution: 1.0
Avg Relevance: 1.0
Avg Accuracy: 1.0
Avg Efficiency: 1.0
Step 0: This step searches for all Python files (files ending with .py) in the system. It is a foundational step that identifies the set of candidate files to evaluate for modification date. Without this step, there would be no list of files to analyze further. The pattern used is accurate and directly relevant to the user's query.
Step 1: This step retrieves file metadata (specifically modification dates) for the identified Python files. This information is essential to determine which files were modified today. The result correctly distinguishes between 'today' and 'yesterday', enabling the final answer to be constructed accurately. This is a critical follow-up to Step 0 and directly supports the user's goal.
```


## Summary

Agent graders provide comprehensive evaluation across all aspects of agent behavior—from individual actions and tool calls to memory management, planning, reflection, and complete trajectories.

**Key capabilities:**

- **Process-level debugging** — Identify specific failure points in tool selection, parameter extraction, or reasoning
- **Outcome-level assessment** — Measure overall task success and trajectory quality
- **Systematic improvement** — Combine multiple graders to diagnose where agents fail, why they fail, and how to improve them

Build complete evaluation pipelines by combining graders from different categories to match your agent architecture and debugging needs.


## Next Steps

- [Multimodal Graders](multimodal.md) — Evaluate image and vision tasks
- [Code & Math Graders](code_math.md) — Evaluate code generation and mathematical problem-solving
- [Build Reward for Training](../get_started/build_reward.md) — Combine multiple graders for RLHF rewards
