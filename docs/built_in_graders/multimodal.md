# Multimodal Graders

Vision-language graders for evaluating AI responses involving images. These graders assess image-text coherence, image helpfulness, and text-to-image generation quality.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `ImageCoherenceGrader` | Measures image-text alignment | LLM-Based | 1-5 | Document generation, content QA |
| `ImageHelpfulnessGrader` | Evaluates image contribution to understanding | LLM-Based | 1-5 | Educational content, tutorials |
| `TextToImageGrader` | Assesses generated image quality | LLM-Based | 1-5 | Text-to-image model evaluation |

## Performance

Benchmark results using `qwen-vl-max` as the judge model:

| Grader | Samples | Preference Accuracy | Avg Score Diff |
|--------|---------|---------------------|----------------|
| **ImageCoherenceGrader** | 20 | 75.00% | 0.23 |
| **ImageHelpfulnessGrader** | 20 | **80.00%** | 0.18 |
| **TextToImageGrader** | 20 | 75.00% | 0.26 |

!!! note "Performance Metrics"
    Preference Accuracy measures alignment with human-annotated preference labels. Higher is better.


## MLLMImage

All multimodal graders use `MLLMImage` to represent images. It supports both URL and base64 formats.

```python
from openjudge.graders.multimodal import MLLMImage

# From URL
image = MLLMImage(url="https://example.com/image.jpg")

# From base64
image = MLLMImage(base64="iVBORw0KGgoAAAANS...", format="png")
```


## ImageCoherenceGrader

Evaluates how well images match and relate to their surrounding text context. Assesses whether images are appropriately placed and meaningfully connected to the content.

**When to use:**
- Document generation with embedded images
- Multimodal content quality assurance
- Educational material evaluation
- Technical documentation review

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | List[str \| MLLMImage] | Yes | Mixed list of text and images |
| `max_context_size` | int | No | Max characters from context (default: 500) |

**What it evaluates:**
- Semantic alignment between image and surrounding text
- Contextual relevance to preceding and following content
- Visual-text consistency
- Placement appropriateness

**Scoring:**
- **5**: Perfect coherence, image perfectly illustrates text
- **4**: Strong coherence with clear relationship
- **3**: Some coherence but connection could be clearer
- **2**: Weak coherence, image seems somewhat misplaced
- **1**: No coherence, image is completely unrelated

!!! note
    Score range is 1-5. For multiple images, returns average score.

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multimodal import ImageCoherenceGrader, MLLMImage

async def main():
    model = OpenAIChatModel(model="qwen-vl-max")
    grader = ImageCoherenceGrader(model=model)

    result = await grader.aevaluate(
        response=[
            "Q3 sales increased by 25% compared to last quarter.",
            MLLMImage(url="https://example.com/sales_chart.jpg"),
            "This growth was primarily driven by new product launches.",
        ]
    )

    print(f"Score: {result.score}")   # 4.8 - image coherent with context
    print(f"Reason: {result.reason}")

asyncio.run(main())
```


## ImageHelpfulnessGrader

Evaluates how helpful images are in aiding readers' understanding of text. Goes beyond simple coherence to assess whether images provide genuine value and clarify concepts.

**When to use:**
- Educational content evaluation
- Technical documentation quality assurance
- Tutorial and how-to guide assessment
- Instructional design evaluation
- User manual review

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | List[str \| MLLMImage] | Yes | Mixed list of text and images |
| `max_context_size` | int | No | Max characters from context (default: 500) |

**What it evaluates:**
- Information enhancement beyond text
- Concept clarification
- Practical utility vs. decorative value
- Educational value
- Comprehension support

**Scoring:**
- **5**: Extremely helpful, significantly enhances understanding
- **4**: Very helpful, provides clear value
- **3**: Somewhat helpful but limited value
- **2**: Minimally helpful
- **1**: Not helpful or redundant with text

!!! note
    Score range is 1-5. For multiple images, returns average score.

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multimodal import ImageHelpfulnessGrader, MLLMImage

async def main():
    model = OpenAIChatModel(model="qwen-vl-max")
    grader = ImageHelpfulnessGrader(model=model)

    result = await grader.aevaluate(
        response=[
            "The system architecture consists of three main layers.",
            MLLMImage(url="https://example.com/architecture_diagram.jpg"),
            "Each layer handles specific responsibilities.",
        ]
    )

    print(f"Score: {result.score}")   # 4.5 - diagram very helpful
    print(f"Reason: {result.reason}")

asyncio.run(main())
```


## TextToImageGrader

Evaluates AI-generated images from text prompts by measuring semantic consistency (prompt following) and perceptual quality (visual realism). Essential for text-to-image model evaluation.

**When to use:**
- Text-to-image model benchmarking (DALL-E, Stable Diffusion, etc.)
- Prompt engineering effectiveness evaluation
- Generative model quality control
- A/B testing different generation parameters

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The text prompt used for generation |
| `response` | MLLMImage | Yes | The generated image to evaluate |

**What it evaluates:**
- **Semantic Consistency**: Image accurately reflects prompt description
- **Element Presence**: All requested elements are included
- **Visual Quality**: Image looks natural and realistic
- **Artifact Detection**: No distortions, blur, or unnatural features
- **Composition**: Proper spatial arrangement and aesthetics

**Scoring:**

The final score combines two dimensions:
- **Semantic Score (1-5)**: How well the image follows the prompt
- **Perceptual Score (1-5)**: Naturalness + artifact absence

Formula: `sqrt(semantic × min(perceptual))` → score in range [1, 5]

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multimodal import TextToImageGrader, MLLMImage

async def main():
    model = OpenAIChatModel(model="qwen-vl-max")
    grader = TextToImageGrader(model=model)

    result = await grader.aevaluate(
        query="A fluffy orange cat sitting on a blue velvet sofa",
        response=MLLMImage(url="https://example.com/generated_cat.jpg"),
    )

    print(f"Score: {result.score}")   # 4.6 - excellent generation
    print(f"Reason: {result.reason}")

    # Access detailed scores
    print(f"Semantic: {result.metadata['min_sc']}/5")
    print(f"Perceptual: {result.metadata['min_pq']}/5")

asyncio.run(main())
```


## Next Steps

- [Code & Math Graders](code_math.md) — Evaluate code generation and mathematical problem-solving
- [Text Graders](text.md) — Fast, deterministic text comparison using various algorithms
- [Build Reward for Training](../get_started/build_reward.md) — Combine multiple graders for RLHF rewards

