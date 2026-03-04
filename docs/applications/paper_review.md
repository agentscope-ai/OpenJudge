# Paper Review

Automatically review academic papers using LLM-powered evaluation. This end-to-end pipeline performs correctness detection, scholarly review, criticality verification, and bibliography validation.


## Overview

Paper Review automates the academic paper review process, providing comprehensive evaluation across multiple dimensions. Ideal for **pre-submission checks**, **conference paper screening**, **research quality assurance**, and **reference verification**.

!!! tip "Multimodal PDF Support"
    This pipeline leverages native PDF understanding capabilities of modern LLMs, eliminating the need for PDF-to-text conversion and preserving figure, table, and formula context.

The pipeline automates five evaluation phases:

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | `Safety Checks` | Detect jailbreaking attempts and format violations |
| 2 | `CorrectnessGrader` | Identify logical errors, contradictions, and factual issues |
| 3 | `ReviewGrader` | Generate comprehensive scholarly review |
| 4 | `CriticalityGrader` | Verify and classify detected issues by severity |
| 5 | `BibChecker` | Validate bibliography entries against CrossRef |


## Quick Start

Get started with Paper Review in just a few lines of code:

=== "Single Paper Review"

    ```python
    import asyncio
    import os
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig, generate_report

    async def main():
        config = PipelineConfig(
            model_name="gemini-3-pro-preview",  # Recommended
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),  # OpenAI-compatible proxy
            timeout=1500,
            enable_safety_checks=True,
            enable_correctness=True,
            enable_review=True,
            enable_criticality=True,
        )

        pipeline = PaperReviewPipeline(config)

        # Review paper and generate report
        result, report = await pipeline.review_and_report(
            pdf_input="paper.pdf",
            paper_name="My Research Paper",
            output_path="review_report.md",
        )

        print(f"Review Score: {result.review.score}/6")
        print(f"Report saved to: review_report.md")

    asyncio.run(main())
    ```

=== "With Bibliography Verification"

    ```python
    import asyncio
    import os
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig

    async def main():
        config = PipelineConfig(
            model_name="gemini-3-pro-preview",  # Recommended
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),  # OpenAI-compatible proxy
            enable_bib_verification=True,
            crossref_mailto="your-email@example.com",  # For CrossRef API
        )

        pipeline = PaperReviewPipeline(config)

        result = await pipeline.review_paper(
            pdf_input="paper.pdf",
            bib_path="references.bib",
        )

        # Check bibliography verification results
        if result.bib_verification:
            for bib_file, summary in result.bib_verification.items():
                print(f"Bibliography: {bib_file}")
                print(f"  Total references: {summary.total_references}")
                print(f"  Verified: {summary.verified}")
                print(f"  Suspect: {summary.suspect}")
                print(f"  Verification rate: {summary.verification_rate:.1%}")

    asyncio.run(main())
    ```

=== "TeX Package Review"

    Review papers directly from arXiv-style TeX source packages:

    ```python
    import asyncio
    import os
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig

    async def main():
        config = PipelineConfig(
            model_name="gemini-3-pro-preview",  # Recommended
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),  # OpenAI-compatible proxy
            enable_bib_verification=True,
            crossref_mailto="your-email@example.com",
        )

        pipeline = PaperReviewPipeline(config)

        # Process .tar.gz or .zip package
        result = await pipeline.review_tex_package("paper_source.tar.gz")

        # Access TeX package info
        if result.tex_info:
            print(f"Main TeX file: {result.tex_info.main_tex}")
            print(f"Total TeX files: {result.tex_info.total_files}")
            print(f"BibTeX files: {result.tex_info.bib_files}")
            print(f"Figures: {result.tex_info.figures}")

    asyncio.run(main())
    ```


## Recommended Models

For optimal paper review quality, we recommend using advanced reasoning models:

| Model | Provider | Best For |
|-------|----------|----------|
| `gemini-3-pro-preview` | Google | Comprehensive review with excellent multimodal understanding |
| `gpt-5.2` | OpenAI | Deep logical analysis and nuanced critique |

!!! note "Model Characteristics"
    **GPT-5.2 excels at detecting subtle errors and inconsistencies**, making it ideal for rigorous correctness checking. In contrast, **Gemini-3.0-Pro tends to be overly generous with praise and may overlook critical issues** — use it when you need comprehensive coverage but verify findings with a more critical model.

!!! warning "Model Requirements"
    Paper review requires models with strong reasoning capabilities and native PDF/image understanding. Smaller models may miss subtle logical errors or produce superficial reviews.

!!! tip "Use OpenAI-Compatible Proxy Services"
    We recommend using **OpenAI-compatible API proxy services** (such as OpenRouter, Together AI, or other third-party providers) to access various models through a unified interface. This approach offers several benefits:

    - **Unified API**: Access multiple model providers (Google, OpenAI, Anthropic, etc.) through a single OpenAI-compatible endpoint
    - **Cost Optimization**: Many proxy services offer competitive pricing and pay-as-you-go billing
    - **Simplified Integration**: No need to manage multiple API keys and endpoints
    - **Fallback Support**: Easily switch between models without code changes


## Component Guide

### Pipeline Configuration

The `PipelineConfig` controls which evaluation phases to run:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"gpt-4o"` | LLM model identifier |
| `api_key` | str | `""` | API key for the model provider |
| `base_url` | str | `None` | Custom API base URL |
| `temperature` | float | `0.7` | Sampling temperature |
| `timeout` | int | `1500` | Request timeout in seconds |
| `enable_safety_checks` | bool | `True` | Run jailbreaking and format checks |
| `enable_correctness` | bool | `True` | Run correctness detection |
| `enable_review` | bool | `True` | Generate scholarly review |
| `enable_criticality` | bool | `True` | Verify detected issues |
| `enable_bib_verification` | bool | `True` | Validate bibliography |
| `crossref_mailto` | str | `None` | Email for CrossRef API (higher rate limits) |

### Graders

Each grader evaluates a specific aspect of the paper:

#### CorrectnessGrader

Detects logical errors, contradictions, and factual issues.

**Output Score (1-3):**

| Score | Meaning |
|-------|---------|
| 1 | No significant errors detected |
| 2 | Minor issues found |
| 3 | Major errors detected |

```python
from cookbooks.paper_review.graders import CorrectnessGrader

grader = CorrectnessGrader(model)
result = await grader.aevaluate(pdf_data=pdf_base64)
print(f"Score: {result.score}")
print(f"Key issues: {result.metadata.get('key_issues', [])}")
```

#### ReviewGrader

Generates a comprehensive scholarly review.

**Output Score (1-6):** Follows standard academic conference scoring:

| Score | Rating |
|-------|--------|
| 6 | Strong Accept |
| 5 | Accept |
| 4 | Weak Accept |
| 3 | Borderline |
| 2 | Weak Reject |
| 1 | Reject |

```python
from cookbooks.paper_review.graders import ReviewGrader

grader = ReviewGrader(model)
result = await grader.aevaluate(pdf_data=pdf_base64)
print(f"Score: {result.score}/6")
print(f"Review: {result.metadata.get('review_text', '')}")
```

#### CriticalityGrader

Verifies detected issues and classifies them by severity.

**Issue Classification:**

- **Major**: Critical issues affecting paper validity
- **Minor**: Issues that don't invalidate core contributions
- **False Positives**: Initially flagged issues that are actually valid

```python
from cookbooks.paper_review.graders import CriticalityGrader

grader = CriticalityGrader(model)
result = await grader.aevaluate(
    pdf_data=pdf_base64,
    correctness_result=correctness_result,  # From CorrectnessGrader
)
print(f"Major issues: {result.metadata.get('major_issues', [])}")
print(f"Minor issues: {result.metadata.get('minor_issues', [])}")
print(f"False positives: {result.metadata.get('false_positives', [])}")
```

#### BibChecker

Validates bibliography entries against CrossRef database.

```python
from cookbooks.paper_review.processors import BibChecker

checker = BibChecker(mailto="your-email@example.com")
results = checker.check_bib_file("references.bib")

print(f"Total: {results['total_references']}")
print(f"Verified: {results['verified']}")
print(f"Suspect: {results['suspect']}")
```

**Verification Status:**

| Status | Description |
|--------|-------------|
| `verified` | Reference found in CrossRef with matching metadata |
| `suspect` | Reference not found or metadata mismatch |
| `error` | Verification failed (API error, malformed entry) |


## Output Format

### PaperReviewResult

The pipeline returns a structured `PaperReviewResult`:

```python
class PaperReviewResult:
    is_safe: bool                    # Whether paper passed safety checks
    safety_issues: List[str]         # Detected safety issues
    correctness: CorrectnessResult   # Correctness detection result
    review: ReviewResult             # Scholarly review result
    criticality: CriticalityResult   # Issue classification result
    format_compliant: bool           # Whether format is acceptable
    bib_verification: Dict[str, BibVerificationSummary]  # Per-file results
    tex_info: TexPackageInfo         # TeX package metadata (if applicable)
    metadata: Dict[str, Any]         # Additional metadata
```

### Markdown Report

Use `generate_report()` to create a human-readable Markdown report:

```python
from cookbooks.paper_review import generate_report

report = generate_report(
    result,
    paper_name="My Research Paper",
    output_path="review_report.md"  # Optional: save to file
)
```

**Report Sections:**

1. **Paper Information** - Basic metadata
2. **Safety Status** - Jailbreaking and format check results
3. **Correctness Analysis** - Detected errors with severity indicators
4. **Scholarly Review** - Full review text and score
5. **Bibliography Verification** - Reference validation summary
6. **TeX Package Info** - Source package details (if applicable)

**Severity Indicators:**

- 🔴 Major issue (critical)
- 🟡 Minor issue (non-critical)
- 🟢 No issues / Verified


## Best Practices

!!! tip "Do"
    - Use **recommended models** (gemini-2.5-pro-preview-05-06 or gpt-5.2) for best results
    - Provide **bibliography files** when available for comprehensive verification
    - Set a **generous timeout** (1500s+) for long papers
    - Include your **email** in `crossref_mailto` for higher API rate limits
    - Review the **criticality classification** to distinguish real issues from false positives

!!! warning "Don't"
    - Use smaller/faster models for paper review (quality will suffer)
    - Skip safety checks in production (prevents prompt injection)
    - Rely solely on automated review (human oversight is essential)
    - Ignore `suspect` references without manual verification

**Performance Tips:**

- For papers > 30 pages, increase `timeout` to 2000+ seconds
- Use TeX package review when source is available (better bibliography extraction)
- Run individual graders separately if you only need specific evaluations


## Examples

Find complete examples in the `cookbooks/paper_review/examples/` directory:

| Example | Description |
|---------|-------------|
| `single_paper_review.py` | Basic PDF review with report generation |
| `bib_verification.py` | Standalone bibliography verification |
| `tex_package_review.py` | Review from TeX source package |
| `correctness_check.py` | Correctness detection only |


**Related Topics:** [Zero-Shot Evaluation](zero_shot_evaluation.md) · [Create Custom Graders](../building_graders/create_custom_graders.md) · [Run Grading Tasks](../running_graders/run_tasks.md)
