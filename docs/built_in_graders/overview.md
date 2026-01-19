# Built-in Graders Overview

 OpenJudge provides **50+ pre-built graders** for evaluating AI responses across quality dimensions, agent behaviors, formats, and modalities. All graders are **rigorously evaluated** on benchmark datasets to ensure reliability and accuracy. For installation, environment setup, and running your first evaluation, see the **[Quick Start Guide](../get_started/quickstart.md)**.


## Key Features

- **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks via specialized graders. Each category provides multiple graders targeting different evaluation dimensions.

- **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire agent lifecycle—including trajectories and specific components such as Memory, Reflection, Tool Use, Planning, and Action Selection.

- **Quality Assurance:** Built for reliability. Every grader comes with benchmark datasets and pytest integration for immediate quality validation. Graders are continuously tested to ensure consistent and accurate results.

- **Unified API Design:** All graders follow a consistent interface with `aevaluate()` method, returning standardized `GraderScore` objects with `score`, `reason`, and `metadata` fields for seamless integration.

- **Flexible Implementation:** Choose between LLM-based graders for nuanced quality assessment or code-based graders for fast, deterministic, zero-cost evaluation depending on your requirements.


## Available Graders

Choose the right grader for your evaluation needs. OpenJudge organizes graders by evaluation focus, making it easy to find graders for specific tasks.

!!! tip "Implementation Types"
    - **LLM-Based** graders: Nuanced quality assessment using LLM judges, suitable for subjective evaluation
    - **Code-Based** graders: Fast, deterministic, zero-cost evaluation using algorithms


### General Graders

Evaluate fundamental response quality including relevance, safety, and correctness. [→ Detailed Documentation](general.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `RelevanceGrader` | Evaluates how relevant a response is to the user's query | LLM-Based | 1-5 |
| `HallucinationGrader` | Detects fabricated information not supported by context | LLM-Based | 1-5 |
| `HarmfulnessGrader` | Identifies harmful, offensive, or inappropriate content | LLM-Based | 1-5 |
| `InstructionFollowingGrader` | Checks if response follows given instructions | LLM-Based | 1-5 |
| `CorrectnessGrader` | Verifies response matches reference answer | LLM-Based | 1-5 |


### Agent Graders

Comprehensive evaluation for AI agents across the entire lifecycle. [→ Detailed Documentation](agent_graders.md)

**Action Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `ActionAlignmentGrader` | Evaluates if agent actions align with goals | LLM-Based | {0, 1} |
| `ActionLoopDetectionGrader` | Detects repetitive action loops | Code-Based | {0, 1} |

**Tool Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `ToolSelectionGrader` | Evaluates appropriateness of tool selection | LLM-Based | 1-5 |
| `ToolCallAccuracyGrader` | Checks tool call correctness | LLM-Based | 1-5 |
| `ToolCallSequenceMatchGrader` | Validates tool call sequence | Code-Based | {0, 1} |
| `ToolCallSuccessGrader` | Checks if tool calls succeeded | LLM-Based | {0, 1} |
| `ToolParameterCheckGrader` | Validates tool parameters | LLM-Based | {0, 1} |

**Memory Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `MemoryAccuracyGrader` | Evaluates accuracy of stored memories | LLM-Based | {0, 1} |
| `MemoryDetailPreservationGrader` | Checks if important details are preserved | LLM-Based | {0, 1} |
| `MemoryRetrievalEffectivenessGrader` | Evaluates memory retrieval quality | LLM-Based | {0, 1} |

**Plan & Reflection Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `PlanFeasibilityGrader` | Evaluates if plans are executable | LLM-Based | {0, 1} |
| `ReflectionAccuracyGrader` | Checks accuracy of agent reflections | LLM-Based | {0, 1} |
| `ReflectionOutcomeUnderstandingGrader` | Evaluates understanding of outcomes | LLM-Based | {0, 1} |
| `ReflectionProgressAwarenessGrader` | Checks awareness of task progress | LLM-Based | {0, 1} |

**Observation Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `ObservationInformationGainGrader` | Evaluates information gain from observations | Code-Based | [0, 1] |

**Trajectory Graders**

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `TrajectoryComprehensiveGrader` | Comprehensive trajectory evaluation | LLM-Based | {0, 1} |


### Text Graders

Fast, deterministic text comparison using various algorithms. [→ Detailed Documentation](text.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `SimilarityGrader` | Text similarity with 15+ algorithms (BLEU, ROUGE, F1, etc.) | Code-Based | [0, 1] |
| `StringMatchGrader` | String matching (exact, prefix, suffix, regex, etc.) | Code-Based | {0, 1} |
| `NumberAccuracyGrader` | Compares numerical values with tolerance | Code-Based | {0, 1} |


### Code Graders

Evaluate code quality, syntax, and execution. [→ Detailed Documentation](code_math.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `CodeExecutionGrader` | Executes code against test cases | Code-Based | [0, 1] |
| `SyntaxCheckGrader` | Validates Python syntax using AST | Code-Based | {0, 1} |
| `CodeStyleGrader` | Checks indentation and naming conventions | Code-Based | [0, 1] |
| `PatchSimilarityGrader` | Compares code patches using SequenceMatcher | Code-Based | [0, 1] |


### Math Graders

Verify mathematical expressions and computations. [→ Detailed Documentation](code_math.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `MathExpressionVerifyGrader` | Verifies math expressions (LaTeX & plain) | Code-Based | {0, 1} |


### Format Graders

Validate structured outputs and formatting. [→ Detailed Documentation](format.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `JsonValidatorGrader` | Validates JSON syntax | Code-Based | {0, 1} |
| `JsonMatchGrader` | Deep comparison of JSON structures | Code-Based | {0, 1} |
| `LengthPenaltyGrader` | Penalizes too short/long responses | Code-Based | ≤0 (penalty) |
| `NgramRepetitionPenaltyGrader` | Penalizes repetitive n-grams | Code-Based | ≤0 (penalty) |
| `ReasoningFormatGrader` | Checks `<think>` and `<answer>` tags | Code-Based | {0, 1} |
| `ReasoningToolCallFormatGrader` | Validates tool call format with JSON | Code-Based | {0, 1} |


### Multimodal Graders

Evaluate vision-language tasks and image quality. [→ Detailed Documentation](multimodal.md)

| Grader | Description | Type | Score Range |
|--------|-------------|------|-------------|
| `ImageCoherenceGrader` | Evaluates image-text coherence | LLM-Based | 1-5 |
| `ImageHelpfulnessGrader` | Assesses if images help understanding | LLM-Based | 1-5 |
| `TextToImageGrader` | Evaluates text-to-image generation quality | LLM-Based | 1-5 |

## Next Steps

**Explore Graders by Category:**

- [General Graders](general.md) — Quality assessment (Relevance, Hallucination, Harmfulness, Instruction Following, Correctness)
- [Agent Graders](agent_graders.md) — Agent evaluation (Action, Tool, Memory, Plan, Reflection, Trajectory)
- [Text Graders](text.md) — Text similarity and matching (15+ algorithms)
- [Code & Math Graders](code_math.md) — Code execution and math verification
- [Format Graders](format.md) — Structure validation (JSON, Length, Repetition, Reasoning Format)
- [Multimodal Graders](multimodal.md) — Vision and image tasks

**Advanced Usage:**

- [Run Evaluation Tasks](../running_graders/run_tasks.md) — Batch processing and reporting
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific evaluators
