# -*- coding: utf-8 -*-
"""Example: rebuttal generation and assessment workflows.

Usage:
    # Generate a rebuttal draft after reviewing a paper
    python -m cookbooks.paper_review.examples.rebuttal_workflow generate \
        --pdf_path paper.pdf --api_key YOUR_KEY

    # Assess an existing rebuttal against a review
    python -m cookbooks.paper_review.examples.rebuttal_workflow assess \
        --pdf_path paper.pdf --rebuttal_path rebuttal.txt --api_key YOUR_KEY

    # Full pipeline: review + generate rebuttal in one call
    python -m cookbooks.paper_review.examples.rebuttal_workflow full \
        --pdf_path paper.pdf --api_key YOUR_KEY
"""

import asyncio
from pathlib import Path

import fire

from cookbooks.paper_review.pipeline import PaperReviewPipeline, PipelineConfig
from cookbooks.paper_review.report import generate_report


async def _generate_rebuttal(
    pdf_path: str,
    model_name: str = "gpt-4o",
    api_key: str = "",
    base_url: str | None = None,
    discipline: str | None = None,
    venue: str | None = None,
    language: str | None = None,
    output_path: str | None = None,
):
    """Review a paper then generate a rebuttal draft with [TODO] placeholders."""
    config = PipelineConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        discipline=discipline,
        venue=venue,
        language=language,
    )
    pipeline = PaperReviewPipeline(config)

    print("Step 1: Reviewing paper...")
    result = await pipeline.review_paper(pdf_path)

    if not result.review:
        print("Review failed or was disabled. Cannot generate rebuttal.")
        return

    print(f"Review score: {result.review.score}/6")
    print("\nStep 2: Generating rebuttal draft...")
    rebuttal = await pipeline.generate_rebuttal(pdf_path, review_result=result)
    result.rebuttal = rebuttal

    report = generate_report(result, Path(pdf_path).stem, output_path)
    if output_path:
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report)

    todo_count = rebuttal.rebuttal_text.count("[TODO:")
    if todo_count:
        print(f"\n>>> {todo_count} [TODO] placeholder(s) require your attention. <<<")


async def _assess_rebuttal(
    pdf_path: str,
    rebuttal_path: str,
    review_text: str | None = None,
    model_name: str = "gpt-4o",
    api_key: str = "",
    base_url: str | None = None,
    discipline: str | None = None,
    language: str | None = None,
    output_path: str | None = None,
):
    """Review a paper then assess an existing rebuttal."""
    config = PipelineConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        discipline=discipline,
        language=language,
    )
    pipeline = PaperReviewPipeline(config)

    rebuttal_text = Path(rebuttal_path).read_text(encoding="utf-8")

    if review_text:
        print("Using provided review text.")
        result = await pipeline.review_paper(pdf_path)
    else:
        print("Step 1: Reviewing paper...")
        result = await pipeline.review_paper(pdf_path)
        if result.review:
            review_text = result.review.review
        else:
            print("Review failed. Cannot assess rebuttal without review text.")
            return

    print("\nStep 2: Assessing rebuttal...")
    assessment = await pipeline.assess_rebuttal(
        pdf_path,
        rebuttal_text=rebuttal_text,
        review_text=review_text,
        review_result=result,
    )
    result.rebuttal_assessment = assessment

    report = generate_report(result, Path(pdf_path).stem, output_path)
    if output_path:
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report)

    print(f"\nScore: {assessment.original_score}/6 -> {assessment.updated_score}/6")


async def _full_pipeline(
    pdf_path: str,
    model_name: str = "gpt-4o",
    api_key: str = "",
    base_url: str | None = None,
    discipline: str | None = None,
    venue: str | None = None,
    language: str | None = None,
    output_path: str | None = None,
):
    """Review + generate rebuttal in a single review_and_report call."""
    config = PipelineConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        discipline=discipline,
        venue=venue,
        language=language,
        enable_rebuttal_generation=True,
    )
    pipeline = PaperReviewPipeline(config)
    result, report = await pipeline.review_and_report(
        pdf_path,
        paper_name=Path(pdf_path).stem,
        output_path=output_path,
    )

    if output_path:
        print(f"Report saved to: {output_path}")
    else:
        print(report)


class RebuttalCLI:
    """CLI for rebuttal workflows."""

    def generate(self, **kwargs):
        """Generate a rebuttal draft."""
        asyncio.run(_generate_rebuttal(**kwargs))

    def assess(self, **kwargs):
        """Assess an existing rebuttal."""
        asyncio.run(_assess_rebuttal(**kwargs))

    def full(self, **kwargs):
        """Full pipeline: review + generate rebuttal."""
        asyncio.run(_full_pipeline(**kwargs))


if __name__ == "__main__":
    fire.Fire(RebuttalCLI)
