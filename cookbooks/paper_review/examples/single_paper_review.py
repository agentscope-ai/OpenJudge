# -*- coding: utf-8 -*-
"""Example: Single paper review with Markdown report generation."""

import asyncio
import os

from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig, generate_report


async def main():
    config = PipelineConfig(
        model_name="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        timeout=1500,
        enable_safety_checks=True,
        enable_correctness=True,
        enable_review=True,
        enable_criticality=True,
    )

    pipeline = PaperReviewPipeline(config)

    # Option 1: Review and get result object
    result = await pipeline.review_paper("paper.pdf")

    # Generate Markdown report
    report = generate_report(result, paper_name="My Paper", output_path="review_report.md")
    print(report)

    # Option 2: Review and generate report in one call
    result, report = await pipeline.review_and_report(
        pdf_input="paper.pdf",
        paper_name="My Paper",
        output_path="review_report.md",
    )
    print("\nReport saved to: review_report.md")


if __name__ == "__main__":
    asyncio.run(main())
