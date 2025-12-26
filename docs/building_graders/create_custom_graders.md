# Create Custom Graders
Custom graders allow you to define precisely how you want to evaluate AI model responses when built-in evaluation tools don't meet your specific needs. This guide helps you build the right grader for your task by following a structured approach to grader design.

!!! tip
    Before creating custom graders, review the [Core Concepts](../get_started/core_concepts.md) to understand how graders fit into the OpenJudge ecosystem.

## Understanding Your Evaluation Needs
Before diving into implementation, it's essential to clearly define what you want to evaluate and in what scenario. Consider whether you're measuring objective properties like length and keyword presence or subjective qualities such as helpfulness and coherence. Determine if you need absolute scores or relative rankings, and think about what constitutes a "good" response in your particular use case.

Depending on your objectives, evaluation can take several forms. For quality assessment, the focus might be on whether responses are factually accurate, effectively address the user's query, maintain a coherent and logical structure, or stay relevant to the topic at hand.

Compliance-focused evaluations serve a different purpose, ensuring that responses adhere to specific guidelines. This could mean verifying the correct format has been used, confirming that the content aligns with safety policies by avoiding harmful material, or simply checking that the model has followed all explicit instructions provided in the prompt.

In contrast, comparative evaluations are designed to rank or select from multiple options. This includes identifying the best-performing model among several candidates, ranking different responses to the same query by quality, or conducting A/B tests to see which version of a prompt yields superior results.

## Choosing the Right Approach
Based on your evaluation needs, you'll need to choose both an evaluation approach (how to structure the evaluation) and an implementation method (how to execute the evaluation).

### Evaluation Approaches
The **Pointwise** approach evaluates each response independently, resulting in a score or classification. It is particularly well-suited for measuring absolute quality, determining if a response meets a specific standard, assessing objective properties, or verifying compliance with fixed rules like formatting or policy guidelines.

Conversely, the **Listwise** approach is inherently comparative. It works by directly comparing multiple responses to the same query, producing a relative ranking. This method is the natural choice when your goal is to select the best candidate from a set of responses or perform a direct head-to-head comparison between models or prompts.

### Implementation Methods
**Code-Based** graders rely on predefined, programmed logic and are most effective for objective assessments. They excel when evaluating quantifiable metrics like response length or keyword presence, where the criteria are clear and unambiguous. Their deterministic nature makes them highly reproducible and cost-effective, especially for high-volume evaluations.

**LLM-Based** graders leverage the language understanding capabilities of large models (such GPT-4 or Qwen) to make nuanced judgments. They are ideal for subjective assessments that require an understanding of context and meaning, such as judging helpfulness, coherence, or overall quality. These graders are also the preferred choice when you need rich, detailed feedback and explanations for their scores.

### Decision Guide
| Scenario | Approach | Method | Why |
| --- | --- | --- | --- |
| Objective properties (length, keywords) | Pointwise | Code-Based | Deterministic, fast, cost-effective |
| Subjective qualities (helpfulness, coherence) | Pointwise | LLM-Based | Handles nuanced judgments |
| Response comparison/selection | Listwise | Either | LLM for quality insight, Code-Based for simplicity |
| High-volume evaluation | Either | Code-Based | Cost-effective at scale |
| Detailed feedback needed | Either | LLM-Based | Rich qualitative output |


You can combine approaches—using both LLM-Based and Code-Based graders—for comprehensive evaluation.

## Implementing Custom Graders
Once you've determined the appropriate approach and implementation method, you can begin developing your custom grader.

### Essential Design Principles

When developing custom graders, ensure they are robust, maintainable, and effective by following core principles:

!!! tip "Core Design Principles"
    1. **Explicit Definitions**: Establish clear input/output definitions and implement proper error handling
    2. **Predictable Scoring**: Use consistent score ranges:
        - Binary outcomes: 0.0 (failure) to 1.0 (success)
        - Graded evaluations: 0-1 or 1-5 scale
        - Rankings: Positive integers starting from 1 (highest rank)

```python
async def evaluate_helpfulness(query: str, response: str) -> GraderScore:
    """Evaluate response helpfulness.

    Args:
        query: The original user query
        response: The model's response to evaluate

    Returns:
        GraderScore with score between 0.0-1.0 and explanation
    """
    try:
        # Your evaluation logic here
        return GraderScore(
            name="helpfulness_evaluator",
            score=calculate_helpfulness_score(query, response),
            reason="Evaluation successful"
        )
    except Exception as e:
        # Return a default score with error information
        return GraderScore(
            name="helpfulness_evaluator",
            score=0.0,
            reason=f"Evaluation failed: {str(e)}"
        )
```

### LLM-Based Grader Implementation

To create effective LLM-Based graders:

!!! info "LLM Grader Components"
    - **Role Definition**: Establish the LLM as an expert evaluator
    - **Clear Instructions**: Provide detailed guidance on what to evaluate and how to score
    - **Scoring Rubric**: Define what each score means
    - **Output Format**: Specify the exact JSON structure for responses

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.openai_chat_model import OpenAIChatModel

# Define your model
model = OpenAIChatModel(
    model="qwen3-32b",
    api_key="your-api-key"
)

# Create your grader with a well-engineered prompt
helpfulness_grader = LLMGrader(
    name="helpfulness_evaluator",
    mode="pointwise",
    model=model,
    template="""
    You are an expert evaluator assessing the helpfulness of AI responses.

    Instructions:
    1. Consider accuracy, completeness, clarity, and relevance
    2. Score 0.0 for completely unhelpful responses
    3. Score 1.0 for exceptionally helpful responses
    4. Score in between for partial helpfulness

    Query: {query}
    Response: {response}

    Provide your response in JSON format:
    {
        "score": <numerical_score_between_0_and_1>,
        "reason": "<brief_explanation_for_score>"
    }
    """,
    description="Evaluates how helpful a response is to the given query"
)
```

!!! tip
    Incorporate examples of good and poor responses when possible to improve consistency.

#### Listwise LLM-Based Example: Response Comparator

For comparative evaluations, you can create graders that directly compare multiple responses:

```python
# Create your comparison grader
comparison_grader = LLMGrader(
    name="response_comparator",
    mode="listwise",
    model=model,
    template="""
    You are an expert judge comparing AI responses to the same query.

    Instructions:
    1. Compare overall quality, considering accuracy and helpfulness
    2. Rank from best (1) to worst (2)
    3. Explain your reasoning briefly

    Query: {query}
    Response 1: {response_1}
    Response 2: {response_2}

    Provide your response in JSON format:
    {
        "rank": [<better_response_number>, <worse_response_number>],
        "reason": "<brief_explanation_for_ranking>"
    }
    """,
    description="Ranks two responses by quality"
)
```

### Code-Based Grader Implementation

!!! info "Code-Based Grader Best Practices"
    Effective Code-Based graders should have:

    - **Transparent Logic**: Clear, understandable evaluation rules
    - **Modular Design**: Separate concerns for maintainability
    - **Edge Case Handling**: Robust error handling
    - **Consistent Scoring**: Predictable score ranges

#### Pointwise Code-Based Example: Content Quality Checker
```python
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore

async def content_quality_checker(query: str, response: str) -> GraderScore:
    """Check content quality based on multiple criteria."""
    # Define quality criteria
    min_length = 20
    required_sections = ["introduction", "body", "conclusion"]

    # Check length
    length_score = min(len(response) / 100.0, 1.0)
    length_pass = len(response) >= min_length

    # Check for required sections
    section_scores = []
    for section in required_sections:
        section_found = section.lower() in response.lower()
        section_scores.append(1.0 if section_found else 0.0)

    section_score = sum(section_scores) / len(required_sections)

    # Calculate overall score
    overall_score = (length_score + section_score) / 2.0

    # Generate reason
    reasons = []
    if length_pass:
        reasons.append(f"Length OK ({len(response)} chars)")
    else:
        reasons.append(f"Too short ({len(response)} chars)")

    found_sections = [sec for i, sec in enumerate(required_sections) if section_scores[i] > 0]
    missing_sections = [sec for i, sec in enumerate(required_sections) if section_scores[i] == 0]

    if found_sections:
        reasons.append(f"Found sections: {', '.join(found_sections)}")
    if missing_sections:
        reasons.append(f"Missing sections: {', '.join(missing_sections)}")

    return GraderScore(
        name="content_quality_checker",
        score=overall_score,
        reason="; ".join(reasons)
    )

# Create the grader
content_quality_grader = FunctionGrader(
    func=content_quality_checker,
    name="content_quality",
    mode="pointwise"
)
```

!!! tip "Advanced Techniques"
    When developing Code-Based graders, consider:

    - **Compiled Regex**: Use for complex pattern matching
    - **Weighted Scoring**: Assign different weights to criteria
    - **Clear Thresholds**: Define explicit pass/fail boundaries
    - **Metric Combination**: Combine multiple simple metrics into complex evaluations

#### Listwise Code-Based Example: Multi-factor Ranker
```python
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderRank

async def multi_factor_ranker(query: str, response_1: str, response_2: str) -> GraderRank:
    """Rank responses based on multiple factors."""

    def calculate_score(response):
        # Factor 1: Length (0-0.3 weight)
        length_score = min(len(response) / 200.0, 1.0) * 0.3

        # Factor 2: Keyword density (0-0.4 weight)
        keywords = ["accurate", "complete", "clear", "relevant"]
        keyword_count = sum(1 for kw in keywords if kw.lower() in response.lower())
        keyword_score = (keyword_count / len(keywords)) * 0.4

        # Factor 3: Structure indicators (0-0.3 weight)
        structure_indicators = [". ", "! ", "? ", "\n\n"]
        structure_count = sum(response.count(indicator) for indicator in structure_indicators)
        structure_score = min(structure_count / 10.0, 1.0) * 0.3

        return length_score + keyword_score + structure_score

    # Calculate scores
    score_1 = calculate_score(response_1)
    score_2 = calculate_score(response_2)

    # Rank based on scores
    if score_1 > score_2:
        rank = [1, 2]
        reason = f"Response 1 scored {score_1:.2f} vs Response 2 scored {score_2:.2f}"
    elif score_2 > score_1:
        rank = [2, 1]
        reason = f"Response 2 scored {score_2:.2f} vs Response 1 scored {score_1:.2f}"
    else:
        rank = [1, 2]  # Tie goes to first response
        reason = f"Both responses scored {score_1:.2f}"

    return GraderRank(
        name="multi_factor_ranker",
        rank=rank,
        reason=reason
    )

# Create the grader
multi_factor_grader = FunctionGrader(
    func=multi_factor_ranker,
    name="multi_factor_ranking",
    mode="listwise"
)
```

## Validating Your Custom Graders
After implementing your custom grader, it's crucial to validate that it effectively measures what you intend to measure and produces reliable results. Proper validation ensures your grader performs as expected and produces meaningful results.

For comprehensive guidance on validating your graders and generating detailed validation reports, please refer to the [Grader Analysis](../running_graders/grader_analysis.md) documentation. This document covers statistical analysis techniques for understanding grader behavior, validation against ground truth data, error analysis to identify specific weaknesses, and building comprehensive validation strategies.

The validation process helps you ensure your grader produces accurate results, measure consistency and reliability, identify potential biases in evaluation, and optimize grader performance based on empirical evidence.

## Running Your Custom Graders
Once you've built and validated your custom graders, you can run them using the [GradingRunner](../running_graders/run_tasks.md). This component orchestrates the execution of multiple graders across your dataset, handles concurrency, transforms data as needed, and organizes the results for analysis.

When running graders, focus on configuring data mappers to connect your dataset fields with grader inputs, setting concurrency levels for optimal performance, combining results with aggregators for comprehensive scoring, and handling errors gracefully to prevent complete task failures.

## Next Steps

- [Generate Graders from Data](generate_graders_from_data.md) — Automate grader creation from labeled examples
- [Run Grading Tasks](../running_graders/run_tasks.md) — Evaluate your models at scale
- [Grader Analysis](../running_graders/grader_analysis.md) — Validate and analyze grader results

