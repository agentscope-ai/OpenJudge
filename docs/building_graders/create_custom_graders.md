# :hammer: Create Custom Graders

When the [built-in graders](../../built-in_graders/overview.md) don't cover your specific evaluation needs, you can create custom graders. This guide shows you how to build both LLM-based and rule-based graders that seamlessly integrate with RM-Gallery's ecosystem.

## Understanding Custom Graders

Custom graders in RM-Gallery fall into two main categories:

1. **LLM-based graders**: For complex, subjective evaluations that require natural language understanding
2. **Rule-based graders**: For deterministic assessments based on predefined criteria

You'll want to create custom graders when you need to:
- Evaluate domain-specific criteria (legal compliance, medical accuracy, etc.)
- Implement unique business logic
- Fine-tune evaluation parameters beyond what built-in graders offer
- Create proprietary evaluation methods

Let's dive into each approach with practical examples.

## :robot: LLM-based Graders

Use LLM-based graders when you need nuanced evaluations that require language understanding. These are perfect for assessing qualities like helpfulness, coherence, or domain expertise.

### Creating a Custom Helpfulness Grader

Here's how to create a custom helpfulness grader step by step:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# First, select your model
model = OpenAIChatModel(
    model="gpt-4",
    api_key="your-api-key"
)

# Then define your grader with a clear template
helpfulness_grader = LLMGrader(
    name="helpfulness_evaluator",
    mode="pointwise",
    model=model,
    template="""
    You are an expert evaluator assessing the helpfulness of AI responses.

    Query: {query}
    Response: {response}

    Rate the helpfulness of the response on a scale of 0.0 to 1.0, where:
    - 0.0 = Not helpful at all
    - 1.0 = Extremely helpful

    Consider factors like accuracy, completeness, clarity, and relevance.

    Provide your response in JSON format:
    {
        "score": <numerical_score>,
        "reason": "<explanation>"
    }
    """,
    description="Evaluates how helpful a response is to the given query"
)

# Now you can use it with GradingRunner
```

This approach gives you full control over the evaluation criteria and scoring methodology.

### Handling LLM Responses

When using LLM-based graders, the LLM generates a natural language response that needs to be parsed into a structured [GraderScore](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/schema.py#L115-L161) or [GraderRank](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/schema.py#L164-L202) object. RM-Gallery provides several methods for handling these responses:

#### Method 1: Automatic JSON Parsing (Recommended)

The simplest approach is to instruct the LLM to return a JSON-formatted response. RM-Gallery will automatically attempt to parse JSON responses:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")

grader = LLMGrader(
    name="structured_evaluator",
    mode="pointwise",
    model=model,
    template="""
    Evaluate the following response to the given query.

    Query: {query}
    Response: {response}

    Provide your evaluation in strict JSON format:
    {{
        "score": <number between 0.0 and 1.0>,
        "reason": "<explanation for the score>"
    }}
    """
)
```

#### Method 2: Structured Output with Pydantic Models

For more robust parsing, you can define a Pydantic model that represents the expected structure of the LLM's response. **Note: This feature requires the underlying model to support structured output (JSON schema enforcement). Not all models support this feature.**

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from pydantic import BaseModel
from typing import List

class DetailedEvaluation(BaseModel):
    score: float
    reason: str
    strengths: List[str]
    weaknesses: List[str]

model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")  # Requires a model that supports structured output

grader = LLMGrader(
    name="structured_model_grader",
    mode="pointwise",
    model=model,
    template="""
    Evaluate the following response to the given query.

    Query: {query}
    Response: {response}

    Provide a detailed evaluation with the following structure:
    - Score (0.0 to 1.0)
    - Reason for the score
    - Strengths (list of strong points)
    - Weaknesses (list of weak points)
    """,
    structured_model=DetailedEvaluation
)
```

#### Method 3: Custom Callback Functions

For more complex parsing logic or when you need to extract additional metadata, you can provide a custom callback function:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
import re
import json

def custom_callback(chat_response) -> dict:
    """Custom callback to extract additional metadata from LLM response.

    Args:
        chat_response: The parsed ChatResponse object from the model

    Returns:
        dict: Dictionary with additional metadata to merge into the result
    """
    response_text = chat_response.content
    if isinstance(response_text, list) and len(response_text) > 0:
        response_text = response_text[0].text if hasattr(response_text[0], 'text') else str(response_text[0])
    else:
        response_text = str(response_text)

    # Extract confidence from response
    confidence_match = re.search(r'"confidence"\s*:\s*(\d+\.?\d*)', response_text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5

    # Try to parse as JSON to extract score and reason
    try:
        parsed = json.loads(response_text)
        extracted_score = parsed.get("score", 0.0)
        extracted_reason = parsed.get("reason", "No reason provided")
    except json.JSONDecodeError:
        # Fallback extraction
        score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', response_text)
        extracted_score = float(score_match.group(1)) if score_match else 0.0
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response_text)
        extracted_reason = reason_match.group(1) if reason_match else "Extracted from text"

    return {
        "confidence": confidence,
        "extracted_score": extracted_score,
        "extracted_reason": extracted_reason,
        "raw_response_length": len(response_text)
    }

model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")

grader = LLMGrader(
    name="custom_callback_grader",
    mode="pointwise",
    model=model,
    template="""
    Evaluate the following response to the given query.

    Query: {query}
    Response: {response}

    Provide your evaluation in JSON format with score, reason, and confidence:
    {{
        "score": <number between 0.0 and 1.0>,
        "reason": "<explanation for the score>",
        "confidence": <confidence in your evaluation from 0.0 to 1.0>
    }}
    """,
    callback=custom_callback
)
```


### Multi-language Support

LLM graders support multi-language templates:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.schema.prompt_template import PromptTemplate
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")

template = PromptTemplate(
    messages={
        "en": [
            ChatMessage(
                role="system",
                content="You are an English evaluator."
            ),
            ChatMessage(
                role="user",
                content="Query: {query}\nResponse: {response}\nRate helpfulness:"
            )
        ],
        "zh": [
            ChatMessage(
                role="system",
                content="你是一个中文评估者。"
            ),
            ChatMessage(
                role="user",
                content="问题: {query}\n回答: {response}\n评估有用性:"
            )
        ]
    }
)

grader = LLMGrader(
    name="multilingual_evaluator",
    mode="pointwise",
    model=model,
    template=template
)
```

## :gear: Rule-based Graders

**Rule-based graders** implement evaluation logic using predefined rules and conditions. There are two approaches to implementing rule-based graders:

### Approach 1: Simple Functions with FunctionGrader

For simple rule-based evaluations, you can use Python functions with the [FunctionGrader](../../core/graders/function_grader.py#L26-L201) class.

#### Simple Length-based Grader

```python
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.schema import GraderScore

async def length_grader(query: str, answer: str) -> GraderScore:
    """Grade based on answer length."""
    length = len(answer)
    # Normalize score to 0-1 range
    score = min(length / 100.0, 1.0)
    return GraderScore(
        name="length_grader",
        score=score,
        reason=f"Answer length: {length} characters"
    )

# Create the grader
grader = FunctionGrader(
    func=length_grader,
    name="length_evaluator",
    mode="pointwise"
)
```

#### Listwise Example with FunctionGrader

```python
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.schema import GraderRank

async def length_ranker(query: str, answer_1: str, answer_2: str) -> GraderRank:
    """Rank answers by length."""
    lengths = [len(answer_1), len(answer_2)]
    # Rank from shortest to longest (1 is shortest)
    rank = [1, 2] if lengths[0] <= lengths[1] else [2, 1]
    return GraderRank(
        name="length_ranker",
        rank=rank,
        reason=f"Answer lengths: {lengths[0]}, {lengths[1]} characters"
    )

# Create listwise grader
grader = FunctionGrader(
    func=length_ranker,
    name="length_ranking",
    mode="listwise"
)
```

### Approach 2: Complex Implementations Extending BaseGrader

For more complex rule-based evaluations, you can extend the [BaseGrader](../../core/graders/base_grader.py#L15-L210) class.

#### Regex Pattern Matching Grader

```python
import re
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderScore

class RegexPatternGrader(BaseGrader):
    """Grader that evaluates responses based on regex pattern matching."""

    def __init__(self, pattern: str, flags: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.pattern = re.compile(pattern, flags)

    async def aevaluate(self, response: str, **kwargs) -> GraderScore:
        """Evaluate if response matches the regex pattern."""
        match = self.pattern.search(response)
        score = 1.0 if match else 0.0
        reason = "Pattern matched" if match else "Pattern not found"

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata={"pattern": self.pattern.pattern}
        )

# Usage
email_grader = RegexPatternGrader(
    name="email_format_checker",
    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
```

#### Composite Rule-based Grader

```python
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderScore

class ContentQualityGrader(BaseGrader):
    """Grader that evaluates content quality based on multiple criteria."""

    def __init__(self, min_length: int = 10, required_keywords: list = None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.required_keywords = required_keywords or []

    async def aevaluate(self, response: str, **kwargs) -> GraderScore:
        """Evaluate content quality based on length and keyword presence."""
        # Check minimum length
        length_check = len(response) >= self.min_length

        # Check required keywords
        keyword_checks = []
        for keyword in self.required_keywords:
            keyword_checks.append(keyword.lower() in response.lower())

        # Calculate score
        checks_passed = sum([length_check] + keyword_checks)
        total_checks = 1 + len(self.required_keywords)
        score = checks_passed / total_checks

        # Generate reason
        reasons = []
        if not length_check:
            reasons.append(f"Too short ({len(response)} chars)")
        else:
            reasons.append(f"Length OK ({len(response)} chars)")

        for i, (keyword, passed) in enumerate(zip(self.required_keywords, keyword_checks)):
            status = "found" if passed else "missing"
            reasons.append(f"Keyword '{keyword}' {status}")

        return GraderScore(
            name=self.name,
            score=score,
            reason="; ".join(reasons)
        )

# Usage
quality_grader = ContentQualityGrader(
    name="content_quality",
    min_length=50,
    required_keywords=["introduction", "conclusion", "evidence"]
)
```

## :clipboard: Best Practices

### Error Handling

Always handle potential errors in your grader implementations:

```python
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderScore

class RobustGrader(BaseGrader):
    async def aevaluate(self, **kwargs) -> GraderScore:
        try:
            # Your evaluation logic here
            result = self.perform_evaluation(**kwargs)
            return result
        except Exception as e:
            # Return a default score with error information
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"Evaluation failed: {str(e)}"
            )
```

### Consistent Return Values

Ensure your graders return consistent score ranges:

- For binary evaluations: Use 0.0 or 1.0
- For graded evaluations: Use a consistent scale (e.g., 0-1 or 1-5)
- For rankings: Use positive integers starting from 1

### Clear Documentation

Provide clear descriptions and examples for your graders:

```python
helpfulness_grader = LLMGrader(
    name="helpfulness",
    description="Rates how helpful a response is on a scale from 0.0 (not helpful) to 1.0 (very helpful)",
    # ... other parameters
)
```

## :white_check_mark: Testing Your Custom Graders

Always test your custom graders with various inputs:

```python
import asyncio

async def test_grader():
    # Test with various inputs
    test_cases = [
        {"query": "What is 2+2?", "response": "4"},
        {"query": "What is 2+2?", "response": "The answer is four."},
        {"query": "What is 2+2?", "response": "I don't know."},
    ]

    grader = YourCustomGrader()

    for case in test_cases:
        result = await grader.aevaluate(**case)
        print(f"Score: {result.score}, Reason: {result.reason}")

# Run the test
asyncio.run(test_grader())
```

Creating custom graders opens up endless possibilities for evaluating AI model outputs according to your specific requirements. Once you've built your graders, you can:

- :rocket: [Run grading tasks](../../running_graders/run_grading_tasks.md) to evaluate your models at scale
- :gear: [Generate graders from data](generate_graders_from_data.md) to automate creation of evaluation criteria
- :chart_with_upwards_trend: [Validate your graders](../../validating_graders/validation_workflow.md) to ensure consistent and reliable results
- :muscle: [Train a grader](train_a_grader/) to build a reward model from your custom evaluations