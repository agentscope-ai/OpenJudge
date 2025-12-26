# Format Graders

Format graders for evaluating structural and formatting aspects of AI responses. These graders validate JSON structures, check length constraints, detect repetition, and verify specific output formats like reasoning tags.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `JsonValidatorGrader` | Validates JSON syntax | Code-Based | {0, 1} | JSON output validation |
| `JsonMatchGrader` | Deep comparison of JSON structures | Code-Based | {0, 1} | API response matching |
| `LengthPenaltyGrader` | Penalizes too short/long responses | Code-Based | ≤0 (penalty) | Control response length |
| `NgramRepetitionPenaltyGrader` | Penalizes repetitive n-grams | Code-Based | ≤0 (penalty) | Detect text repetition |
| `ReasoningFormatGrader` | Checks `<think>` and `<answer>` tags | Code-Based | {0, 1} | Chain-of-thought format |
| `ReasoningToolCallFormatGrader` | Validates tool call format with JSON | Code-Based | {0, 1} | Agent tool calls |


## JSON Validation

This category provides graders for validating JSON syntax and comparing JSON structures.

### JsonValidatorGrader

Validates whether a response is valid JSON, ensuring structured outputs can be parsed correctly. Use this grader when you need to verify structured data generation, validate API responses, or enforce JSON output requirements in your AI systems.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate as JSON |

**Scoring:**
- **1.0**: Valid JSON that can be parsed
- **0.0**: Invalid JSON or parse error

**Example:**

```python
import asyncio
from openjudge.graders.format.json.json_validator import JsonValidatorGrader

async def main():
    grader = JsonValidatorGrader()

    # Valid JSON
    result = await grader.aevaluate(
        response='{"name": "Alice", "age": 30, "skills": ["Python", "AI"]}',
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid JSON"

    # Invalid JSON
    result = await grader.aevaluate(
        response='{"name": "Alice", "age": 30',  # Missing closing brace
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # Error message

asyncio.run(main())
```


### JsonMatchGrader

Performs deep structural comparison of JSON objects by recursively validating that two JSON structures match according to configurable rules. This grader is ideal for comparing generated JSON outputs against ground truth, verifying API responses, evaluating structured data accuracy, and testing JSON generation quality.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reference_response` | str | Yes | Reference JSON string |
| `response` | str | Yes | Generated JSON to compare |
| `strict_order` | bool | No | Whether list order matters (default: True) |
| `ignore_extra_keys` | bool | No | Ignore extra keys in response (default: False) |

**Scoring:**
- **1.0**: JSON structures match completely
- **0.0**: Structures differ or parse error

**Example:**

```python
import asyncio
from openjudge.graders.format.json.json_match import JsonMatchGrader

async def main():
    # Strict matching
    grader = JsonMatchGrader(strict_order=True)

    result = await grader.aevaluate(
        reference_response='{"name": "Alice", "hobbies": ["reading", "swimming"]}',
        response='{"name": "Alice", "hobbies": ["reading", "swimming"]}',
    )

    print(f"Score: {result.score}")   # 1.0 - exact match

    # Order-independent matching
    grader = JsonMatchGrader(strict_order=False)

    result = await grader.aevaluate(
        reference_response='{"hobbies": ["reading", "swimming"]}',
        response='{"hobbies": ["swimming", "reading"]}',
    )

    print(f"Score: {result.score}")   # 1.0 - matches despite different order

    # Ignore extra keys
    grader = JsonMatchGrader(ignore_extra_keys=True)

    result = await grader.aevaluate(
        reference_response='{"name": "Alice"}',
        response='{"name": "Alice", "age": 30, "city": "NYC"}',
    )

    print(f"Score: {result.score}")   # 1.0 - extra keys ignored

asyncio.run(main())
```


## Length & Quality Control

This category provides graders for controlling output length and detecting repetitive patterns.

### LengthPenaltyGrader

Applies penalties to responses that are too short or too long, helping you control output verbosity. This grader enforces response length constraints by penalizing overly verbose outputs while ensuring minimum content length, making it valuable for training models to generate concise yet complete responses.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to evaluate |
| `min_length` | int | No | Minimum acceptable length (default: 10) |
| `max_length` | int | No | Maximum acceptable length (default: 1000) |
| `penalty_rate` | float | No | Penalty per character violation (default: 0.01) |

**Scoring:**
- **0.0**: Length within acceptable range
- **< 0.0**: Negative penalty proportional to length violation

**Penalty calculation:**
- If `length < min_length`: penalty = -(min_length - length) × penalty_rate
- If `length > max_length`: penalty = -(length - max_length) × penalty_rate

**Example:**

```python
import asyncio
from openjudge.graders.format.length_penalty import LengthPenaltyGrader

async def main():
    grader = LengthPenaltyGrader(
        min_length=50,
        max_length=200,
        penalty_rate=0.1,
    )

    # Acceptable length
    result = await grader.aevaluate(
        response="This response has an acceptable length that falls within the specified range.",
    )

    print(f"Score: {result.score}")   # 0.0 - no penalty
    print(f"Reason: {result.reason}") # "Length acceptable: 50 <= 83 <= 200"

    # Too short
    result = await grader.aevaluate(response="Short")

    print(f"Score: {result.score}")   # -4.5 = -(50-5) * 0.1
    print(f"Reason: {result.reason}") # "Too short: 5 < 50"

    # Too long
    long_text = "A" * 250
    result = await grader.aevaluate(response=long_text)

    print(f"Score: {result.score}")   # -5.0 = -(250-200) * 0.1
    print(f"Reason: {result.reason}") # "Too long: 250 > 200"

asyncio.run(main())
```


### NgramRepetitionPenaltyGrader

Detects and penalizes repetitive patterns in text using N-gram analysis with support for multiple languages and tokenization methods. This grader is essential for quality control of generated text, helping you detect repetitive content, train models to avoid repetition, and evaluate overall text diversity.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to analyze |
| `n` | int | No | N-gram size (default: 3) |
| `penalty_threshold` | float | No | Threshold for hard penalty (default: 0.3) |
| `penalty_rate` | float | No | Penalty rate per repetition (default: 1.0) |
| `use_soft_penalty` | bool | No | Use soft penalty mode (default: False) |
| `max_penalty` | float | No | Maximum penalty value (default: -1.0) |
| `tokenizer_type` | str | No | Tokenizer type: tiktoken, jieba, simple (default: tiktoken) |
| `analyze_scope` | str | No | Analyze "thought" or "full" text (default: full) |

**Scoring:**
- **0.0**: No significant repetition detected
- **< 0.0**: Negative penalty proportional to repetition rate

**Example:**

```python
import asyncio
from openjudge.graders.format.ngram_repetition_penalty import NgramRepetitionPenaltyGrader

async def main():
    # Hard threshold penalty
    grader = NgramRepetitionPenaltyGrader(
        n=3,
        penalty_threshold=0.3,
        penalty_rate=1.0,
    )

    # Diverse text
    result = await grader.aevaluate(
        response="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    )

    print(f"Score: {result.score}")   # 0.0 or small penalty
    print(f"Metadata: {result.metadata['repetition_rate']}")

    # Repetitive text
    result = await grader.aevaluate(
        response="This is a test. This is a test. This is a test. This is a test.",
    )

    print(f"Score: {result.score}")   # Large negative penalty
    print(f"Repetition rate: {result.metadata['repetition_rate']:.2f}")

    # Soft penalty mode
    grader = NgramRepetitionPenaltyGrader(
        n=2,
        use_soft_penalty=True,
        max_penalty=-2.0,
        min_scaling=0.2,
    )

    result = await grader.aevaluate(
        response="Different words create different patterns without repetition here.",
    )

    print(f"Score: {result.score}")   # Gradual penalty

asyncio.run(main())
```


## Reasoning Format Validation

This category provides graders for validating structured reasoning outputs and agent tool calls.

### ReasoningFormatGrader

Validates that responses follow a specific reasoning format with `<think>` and `<answer>` tags, essential for chain-of-thought evaluation. Use this grader to enforce structured reasoning in your models, validate chain-of-thought (CoT) formatting, and ensure proper separation between the thinking process and final answers.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate |
| `think_token` | str | No | Thinking tag name (default: "think") |
| `answer_token` | str | No | Answer tag name (default: "answer") |

**Scoring:**
- **1.0**: Both `<think>` and `<answer>` tags present
- **0.0**: Missing one or both required tags

**Example:**

```python
import asyncio
from openjudge.graders.format.reasoning_format import ReasoningFormatGrader

async def main():
    grader = ReasoningFormatGrader()

    # Valid format
    result = await grader.aevaluate(
        response="""<think>
First, I need to analyze the problem.
The user is asking about Python benefits.
</think>

<answer>
Python is easy to learn, has extensive libraries, and strong community support.
</answer>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "All format requirements met"

    # Invalid format - missing tags
    result = await grader.aevaluate(
        response="Python is a great programming language for beginners.",
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Missing <think></think> tags; Missing <answer></answer> tags"

    # Custom tags
    grader = ReasoningFormatGrader(think_token="reasoning", answer_token="solution")

    result = await grader.aevaluate(
        response="<reasoning>My thought process</reasoning>\n<solution>Final answer</solution>",
    )

    print(f"Score: {result.score}")   # 1.0

asyncio.run(main())
```


### ReasoningToolCallFormatGrader

Validates that responses follow proper format for tool-calling agents with reasoning by checking for `<think>` tags combined with either `<answer>` or `<tool_call>` tags, and validating JSON structure in tool calls. This grader is ideal for agent output validation, enforcing tool-calling formats, verifying function calls, and ensuring proper multi-step reasoning with tool use.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate |

**Valid formats:**
1. `<think>...</think>` + `<answer>...</answer>` - Reasoning with final answer
2. `<think>...</think>` + `<tool_call>...</tool_call>` - Reasoning with tool calls

**Tool call JSON requirements:**
- Must contain `name` field (function name)
- Must contain `arguments` field (function arguments)

**Scoring:**
- **1.0**: Valid format with proper tags and JSON structure
- **0.0**: Invalid format, missing tags, or malformed JSON

**Example:**

```python
import asyncio
from openjudge.graders.format.reasoning_tool_format import ReasoningToolCallFormatGrader

async def main():
    grader = ReasoningToolCallFormatGrader()

    # Valid reasoning + answer format
    result = await grader.aevaluate(
        response="""<think>
The user wants to know the weather. I should provide the current information.
</think>

<answer>
The current temperature is 72°F with clear skies.
</answer>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid <think></think> + <answer></answer> format"

    # Valid reasoning + tool call format
    result = await grader.aevaluate(
        response="""<think>
I need to search for information about Python.
</think>

<tool_call>
{"name": "search", "arguments": {"query": "Python programming language"}}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid <think></think> + <tool_call></tool_call> format with valid JSON"

    # Multiple tool calls
    result = await grader.aevaluate(
        response="""<think>
I need to gather data from multiple sources.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "New York"}}
</tool_call>

<tool_call>
{"name": "get_news", "arguments": {"topic": "technology"}}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Tool calls: {result.metadata['tool_call_count']}")  # 2

    # Invalid format - missing think tag
    result = await grader.aevaluate(
        response="<answer>Direct answer without thinking</answer>",
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Missing <think></think> tags"

    # Invalid format - malformed JSON in tool call
    result = await grader.aevaluate(
        response="""<think>Searching</think>
<tool_call>
{invalid json}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Invalid JSON format in <tool_call> tags"

asyncio.run(main())
```


## Next Steps

- [Building Graders Overview](../building_graders/overview.md) — Learn how to create custom graders
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific format validators
- [Run Grading Tasks](../running_graders/run_tasks.md) — Execute graders at scale with GradingRunner






