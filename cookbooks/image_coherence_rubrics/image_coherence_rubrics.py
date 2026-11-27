#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-Text Coherence with Generated Rubrics

Demonstrates the generate-once-reuse-everywhere pattern for rubrics:

    Step 1: Generate evaluation rubrics ONCE from a shared task description
            with TaskBasedRubricGenerator.
    Step 2: Pass the generated criteria into ImageCoherenceGrader(rubrics=...)
            and judge every candidate against the SAME criteria.
    Step 3: Compare the generated rubrics against the grader's built-in
            rubric on the same samples. When human labels are available,
            report each variant's Pearson correlation with the human scores
            (via CorrelationAnalyzer).

Why generate once and reuse? Rubric generation costs one extra LLM call per
task, not per candidate. Reusing one rubric also keeps every candidate on
identical criteria, which is what makes scores comparable across candidates.

Usage:
    export OPENAI_API_KEY=your_api_key
    export OPENAI_BASE_URL=https://your-api-endpoint  # optional

    python image_coherence_rubrics.py

The example image URLs are placeholders — swap in your own images and human
labels to reproduce the agreement comparison on your data.
"""

import asyncio
import os
from typing import Any, Dict, List

from openjudge.analyzer.validation import CorrelationAnalyzer
from openjudge.generator.simple_rubric import TaskBasedRubricGenerator
from openjudge.graders.multimodal import ImageCoherenceGrader, MLLMImage
from openjudge.graders.schema import GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel

# =============================================================================
# Shared task description — the single source the rubric is generated from
# =============================================================================

TASK_DESCRIPTION = (
    "Evaluate how coherently an illustration image fits into a document. "
    "The image content must match the topic of the surrounding paragraphs, "
    "and its placement should support the narrative instead of interrupting it."
)
SCENARIO = "Automated document illustration review for a content team"

# =============================================================================
# Example samples — one shared candidate set judged by both rubric variants.
# `human_score` is an optional 1-5 label from a human reviewer; it powers the
# agreement comparison in step 3. Replace images and labels with your own.
# =============================================================================

SAMPLES: List[Dict[str, Any]] = [
    {
        "context_above": "Quarterly revenue grew 25% year over year, driven by subscriptions.",
        "image_url": "https://example.com/images/revenue_chart.png",
        "context_below": "Figure 1 breaks the growth down by region.",
        "human_score": 5,
    },
    {
        "context_above": "Our onboarding flow takes under three minutes to complete.",
        "image_url": "https://example.com/images/legacy_architecture.png",
        "context_below": "Figure 2 shows the current signup screens.",
        "human_score": 2,
    },
    {
        "context_above": "The manufacturing process relies on three precision stages.",
        "image_url": "https://example.com/images/assembly_line.png",
        "context_below": "Each stage is described in the sections that follow.",
        "human_score": 4,
    },
    {
        "context_above": "Customer support response times improved sharply in May.",
        "image_url": "https://example.com/images/office_photo.png",
        "context_below": "The remaining sections cover hiring plans.",
        "human_score": 2,
    },
]


def _response_for(sample: Dict[str, Any]) -> list:
    """Mixed list the multimodal graders expect: text, image, text."""
    return [sample["context_above"], MLLMImage(url=sample["image_url"]), sample["context_below"]]


async def _generate_rubrics(model: OpenAIChatModel) -> List[str]:
    """Generate the shared rubric once from the task description.

    ``sample_queries`` goes to ``generate()`` — it gives the LLM a feel for
    the queries that will be evaluated, not to the constructor.
    """
    generator = TaskBasedRubricGenerator(
        model=model,
        task_description=TASK_DESCRIPTION,
        scenario=SCENARIO,
    )
    return await generator.generate(sample_queries=[s["context_above"] for s in SAMPLES])


def _pearson_note() -> str:
    """One-line pointer to the analyzer used for the agreement comparison."""
    return "Agreement = Pearson correlation between grader scores and human labels (CorrelationAnalyzer)."


async def main() -> None:
    """Generate rubrics once, judge every candidate against them, compare
    the generated rubric with the built-in one against human labels."""
    model = OpenAIChatModel(model="gpt-4o-mini", stream=False)

    # ------------------------------------------------------------------
    # Step 1: generate the rubric ONCE from the shared task description.
    # One LLM call total — not one per candidate.
    # ------------------------------------------------------------------
    rubrics = await _generate_rubrics(model)

    print("=" * 72)
    print("Generated rubrics (single source of truth for every candidate):")
    for i, rubric in enumerate(rubrics, 1):
        print(f"  {i}. {rubric}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 2: judge every candidate against the SAME generated criteria,
    # and against the grader's built-in rubric for comparison.
    # ------------------------------------------------------------------
    default_grader = ImageCoherenceGrader(model=model)  # built-in rubric
    custom_grader = ImageCoherenceGrader(model=model, rubrics=rubrics)  # generated

    default_results: List[GraderScore] = []
    custom_results: List[GraderScore] = []
    for sample in SAMPLES:
        response = _response_for(sample)
        default_results.append(await default_grader.aevaluate(response=response))
        custom_results.append(await custom_grader.aevaluate(response=response))

    # ------------------------------------------------------------------
    # Step 3: side-by-side scores, then agreement with the human labels.
    # ------------------------------------------------------------------
    print(f"\n{'sample':<8}{'human':<8}{'built-in':<10}{'generated':<11}")
    for i, sample in enumerate(SAMPLES):
        human = sample.get("human_score", "-")
        print(f"{i + 1:<8}{human:<8}{default_results[i].score:<10.2f}{custom_results[i].score:<11.2f}")

    has_human_labels = all("human_score" in s for s in SAMPLES)
    if has_human_labels:
        dataset = [{"label": s["human_score"]} for s in SAMPLES]
        analyzer = CorrelationAnalyzer()
        print("\n" + _pearson_note())
        for name, results in [("built-in rubric", default_results), ("generated rubrics", custom_results)]:
            analysis = analyzer.analyze(dataset, results)
            print(f"  {name:<18} correlation with human labels: {analysis.correlation:.3f}")
        print(
            "\nInterpretation: the variant with the higher correlation tracks human "
            "judgments more closely. Run this on a larger labeled sample of your own "
            "data before drawing conclusions."
        )
    else:
        print("\nAdd 'human_score' labels to the samples to enable the agreement comparison.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) before running.")
    else:
        asyncio.run(main())
