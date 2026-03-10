# Graders Reference

Graders are the core evaluation units in OpenJudge.
Every grader inherits from `BaseGrader` and implements `async _aevaluate(**kwargs)`.

## Grader Types

| Type | Class | Best for |
|------|-------|----------|
| LLM-based | `LLMGrader` | Subjective quality, semantic understanding |
| Function-based | `FunctionGrader` | Exact rules, fast deterministic checks |
| Agentic | `AgenticGrader` | Evaluation requiring tool calls (search, code run) |

---

## Built-in Graders — Quick Reference

### `common/` — General-purpose (all LLM-based, POINTWISE, score 1–5)

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `CorrectnessGrader` | `openjudge.graders.common.correctness` | `query`, `response`, `reference_response`, `context` | Factual match against reference |
| `HallucinationGrader` | `openjudge.graders.common.hallucination` | `query`, `response`, `context` | Fabricated/unsupported claims |
| `RelevanceGrader` | `openjudge.graders.common.relevance` | `query`, `response` | How relevant the response is |
| `HarmfulnessGrader` | `openjudge.graders.common.harmfulness` | `query`, `response` | Toxic or harmful content |
| `InstructionFollowingGrader` | `openjudge.graders.common.instruction_following` | `query`, `response` | Instruction compliance |
| `SearchCorrectnessGrader` | `openjudge.graders.common.search_correctness` | `query`, `response`, `context` | Correctness in RAG/search context |

All `common/` graders accept `model` (required) and optional `threshold`, `language`, `strategy`.

```python
from openjudge.graders.common.hallucination import HallucinationGrader

grader = HallucinationGrader(model=model)
result = await grader.aevaluate(
    query="Who invented the telephone?",
    response="Thomas Edison invented the telephone in 1876.",
    context="Alexander Graham Bell is credited with the telephone (1876).",
)
# result.score: 1–5  (5 = no hallucination, 1 = severe hallucination)
```

---

### `text/` — String & Text Matching (no LLM needed)

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `StringMatchGrader` | `openjudge.graders.text.string_match` | `response`, `reference`, `algorithm` | Exact/regex/overlap matching |
| `SimilarityGrader` | `openjudge.graders.text.similarity` | `response`, `reference` | ROUGE / BM25 / embedding similarity |
| `NumberAccuracyGrader` | `openjudge.graders.text.number_accuracy` | `response`, `reference` | Numerical value accuracy |

**StringMatchGrader algorithms:** `exact_match`, `prefix_match`, `suffix_match`, `regex_match`,
`substring_match`, `contains_all`, `contains_any`, `word_overlap`, `char_overlap`

```python
from openjudge.graders.text.string_match import StringMatchGrader

grader = StringMatchGrader()
result = await grader.aevaluate(
    response="The capital is Paris.",
    reference="Paris",
    algorithm="substring_match",
)
# result.score: 1.0 (match) or 0.0 (no match)
```

---

### `code/` — Code Evaluation

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `CodeExecutionGrader` | `openjudge.graders.code.code_execution` | `response`, `test_cases` | Test case pass rate |
| `SyntaxCheckGrader` | `openjudge.graders.code.syntax_checker` | `response` | Syntax validity |
| `CodeStyleGrader` | `openjudge.graders.code.code_style` | `response` | Style/lint quality |
| `PatchSimilarityGrader` | `openjudge.graders.code.patch_similarity` | `response`, `reference` | Patch/diff similarity |

```python
from openjudge.graders.code.code_execution import CodeExecutionGrader

grader = CodeExecutionGrader(timeout=10)
result = await grader.aevaluate(
    response="def add(a, b): return a + b",
    test_cases=[{"input": [1, 2], "expected_output": 3}],
)
# result.score: fraction of passed test cases (0.0–1.0)
```

---

### `format/` — Output Format Validation

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `JsonValidatorGrader` | `openjudge.graders.format.json.json_validator` | `response` | Is response valid JSON? |
| `JsonMatchGrader` | `openjudge.graders.format.json.json_match` | `response`, `reference` | JSON structure/content match |
| `LengthPenaltyGrader` | `openjudge.graders.format.length_penalty` | `response` | Penalizes over/under-length |
| `NgramRepetitionPenaltyGrader` | `openjudge.graders.format.ngram_repetition_penalty` | `response` | Penalizes repeated n-grams |
| `ReasoningFormatGrader` | `openjudge.graders.format.reasoning_format` | `response` | `<think>...</think>` format check |

```python
from openjudge.graders.format.json.json_validator import JsonValidatorGrader

grader = JsonValidatorGrader()
result = await grader.aevaluate(response='{"key": "value"}')
# result.score: 1.0 (valid JSON) or 0.0 (invalid)
```

---

### `math/` — Mathematical Expressions

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `MathExpressionVerifyGrader` | `openjudge.graders.math.math_expression_verify` | `response`, `reference` | Mathematical equivalence |

---

### `agent/` — Agent Behavior Evaluation (all LLM-based)

| Category | Class | What it measures |
|----------|-------|-----------------|
| **Tool** | `ToolCallAccuracyGrader` | Whether tool calls are correct |
| **Tool** | `ToolCallSuccessGrader` | Whether tool calls succeeded |
| **Tool** | `ToolSelectionGrader` | Whether the right tool was chosen |
| **Tool** | `ToolParameterCheckGrader` | Correctness of tool parameters |
| **Tool** | `ToolCallStepSequenceMatchGrader` | Tool call order vs expected |
| **Tool** | `ToolCallPrecisionRecallMatchGrader` | Precision/recall of tool call set |
| **Memory** | `MemoryAccuracyGrader` | Accuracy of stored memory |
| **Memory** | `MemoryDetailPreservationGrader` | Detail retention in memory |
| **Memory** | `MemoryRetrievalEffectivenessGrader` | Quality of memory retrieval |
| **Plan** | `PlanFeasibilityGrader` | Whether the plan is feasible |
| **Reflection** | `ReflectionAccuracyGrader` | Accuracy of self-reflection |
| **Action** | `ActionAlignmentGrader` | Action alignment with intent |
| **Trajectory** | `TrajectoryAccuracyGrader` | Trajectory vs reference |
| **Trajectory** | `TrajectoryComprehensiveGrader` | End-to-end trajectory quality |

```python
from openjudge.graders.agent import ToolCallAccuracyGrader

grader = ToolCallAccuracyGrader(model=model)
result = await grader.aevaluate(
    query="Search for today's weather",
    tool_calls=[{"name": "web_search", "arguments": {"query": "today weather"}}],
    reference_tool_calls=[{"name": "web_search", "arguments": {"query": "weather today"}}],
)
```

---

### `multi_turn/` — Multi-turn Conversation (all LLM-based)

| Class | What it measures |
|-------|-----------------|
| `ContextMemoryGrader` | Recalls details from early turns |
| `AnaphoraResolutionGrader` | Pronoun/reference resolution |
| `TopicSwitchGrader` | Handles sudden topic changes |
| `SelfCorrectionGrader` | Corrects errors when given feedback |
| `InstructionClarificationGrader` | Asks for clarification when needed |
| `ProactiveInteractionGrader` | Proactively engages in conversation |
| `ResponseRepetitionGrader` | Avoids repeating prior content |

```python
from openjudge.graders.multi_turn import ContextMemoryGrader

grader = ContextMemoryGrader(model=model)
result = await grader.aevaluate(
    conversation_history=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"},
    ],
    response="Your name is Alice.",
)
```

---

### `multimodal/` — Vision & Image (requires VL model)

| Class | Import | What it measures |
|-------|--------|-----------------|
| `TextToImageGrader` | `openjudge.graders.multimodal.text_to_image` | Text-image alignment |
| `ImageCoherenceGrader` | `openjudge.graders.multimodal.image_coherence` | Image sequence coherence |
| `ImageHelpfulnessGrader` | `openjudge.graders.multimodal.image_helpfulness` | Image usefulness for context |

```python
from openjudge.models.qwen_vl_model import QwenVLModel
from openjudge.graders.multimodal.text_to_image import TextToImageGrader

vl_model = QwenVLModel(model="qwen-vl-plus", api_key="sk-xxx")
grader = TextToImageGrader(model=vl_model)
result = await grader.aevaluate(
    prompt="A red apple on a wooden table",
    image_url="https://example.com/image.jpg",
)
```

---

## LLMGrader — Custom Prompt Grader

Use `LLMGrader` directly when no built-in grader fits. Provide a template string with
`{placeholder}` variables that match your `aevaluate()` kwargs.

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode

grader = LLMGrader(
    model=model,
    name="helpfulness",
    mode=GraderMode.POINTWISE,
    template="""
You are an evaluation assistant.

Query: {query}
Response: {response}

Rate the helpfulness of the response on a scale of 0.0 to 1.0.
Respond in JSON: {{"score": <float>, "reason": "<explanation>"}}
""",
)

result = await grader.aevaluate(
    query="How do I reverse a list in Python?",
    response="Use list.reverse() or reversed().",
)
# result.score, result.reason
```

### Listwise (ranking) mode

```python
ranking_grader = LLMGrader(
    model=model,
    name="quality_rank",
    mode=GraderMode.LISTWISE,
    template="""
Rank the following responses to the query from best (1) to worst.

Query: {query}
Response 1: {response_1}
Response 2: {response_2}

Respond in JSON: {{"rank": [<int>, <int>], "reason": "<explanation>"}}
""",
)

result = await ranking_grader.aevaluate(
    query="Explain gravity",
    response_1="Gravity is a fundamental force...",
    response_2="Things fall down.",
)
# result.rank e.g. [1, 2]  → response_1 is better
```

---

## FunctionGrader — Pure Python Evaluation

Use when the scoring logic is deterministic and requires no LLM.

```python
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore, GraderMode

def length_check(response: str, min_words: int = 10) -> GraderScore:
    word_count = len(response.split())
    score = 1.0 if word_count >= min_words else word_count / min_words
    return GraderScore(
        name="length_check",
        score=score,
        reason=f"Response has {word_count} words (min: {min_words})",
    )

grader = FunctionGrader(
    func=length_check,
    name="length_check",
    mode=GraderMode.POINTWISE,
    min_words=20,          # extra kwargs passed to func
)
result = await grader.aevaluate(response="Short answer.")
```

### Decorator syntax

```python
@FunctionGrader.wrap
def exact_match(response: str, reference: str) -> GraderScore:
    score = 1.0 if response.strip() == reference.strip() else 0.0
    return GraderScore(name="exact_match", score=score, reason="")

grader = exact_match(name="exact_match", mode=GraderMode.POINTWISE)
```

---

## AgenticGrader — Tool-augmented Evaluation

Use when the evaluation itself requires external tools (e.g., web search to verify facts).

```python
from openjudge.agentic import ReActAgent
from openjudge.graders.agentic_grader import AgenticGrader

# Step 1: build agent with tools
agent = ReActAgent(
    model={"model": "gpt-4o", "api_key": "sk-..."},
    tools=[WebSearchTool()],      # any BaseTool implementation
    max_iterations=10,
)

# Step 2: create grader
grader = AgenticGrader(
    agent=agent,
    name="fact_check",
    template="""
Verify the factual accuracy of the response using web search if needed.

Query: {query}
Response: {response}

Return JSON: {{"score": <0.0-1.0>, "reason": "<explanation>"}}
""",
)

result = await grader.aevaluate(
    query="When was Python first released?",
    response="Python was first released in 1991.",
)
```

---

## Custom Grader — Extend BaseGrader

```python
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore

class KeywordGrader(BaseGrader):
    def __init__(self, keywords: list[str], **kwargs):
        super().__init__(name="keyword_grader", mode=GraderMode.POINTWISE, **kwargs)
        self.keywords = keywords

    async def _aevaluate(self, response: str, **kwargs) -> GraderScore:
        hits = sum(1 for kw in self.keywords if kw.lower() in response.lower())
        score = hits / len(self.keywords)
        return GraderScore(
            name=self.name,
            score=score,
            reason=f"{hits}/{len(self.keywords)} keywords found",
        )

    @staticmethod
    def get_metadata():
        return {"description": "Checks keyword presence in response"}

grader = KeywordGrader(keywords=["Python", "list", "reverse"])
result = await grader.aevaluate(response="Use list.reverse() in Python.")
```
