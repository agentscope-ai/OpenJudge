# -*- coding: utf-8 -*-
"""Example: TeX package processing and review."""

import asyncio
import os

from cookbooks.paper_review.pipeline import PaperReviewPipeline, PipelineConfig
from cookbooks.paper_review.processors.tex_processor import TexPackageProcessor


def demo_tex_processor():
    """Demonstrate TeX package processing."""
    print("=" * 60)
    print("TEX PACKAGE PROCESSOR DEMO")
    print("=" * 60)

    processor = TexPackageProcessor()

    # Process a TeX package (replace with actual path)
    # package = processor.process_package("paper_source.tar.gz")

    # For demonstration, show what would happen:
    print(
        """
Usage:
    processor = TexPackageProcessor()
    package = processor.process_package("paper_source.tar.gz")

    print(f"Main TeX file: {package.main_tex}")
    print(f"Total TeX files: {len(package.files)}")
    print(f"BibTeX files: {[b.path for b in package.bib_files]}")
    print(f"Figures: {len(package.figure_paths)} files")
    print(f"Merged content length: {len(package.merged_content)} chars")

    # Access bib file contents directly
    for bib in package.bib_files:
        print(f"\\n{bib.path}:")
        print(bib.content[:200] + "...")
"""
    )


async def demo_full_pipeline():
    """Demonstrate full pipeline with TeX package."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMO")
    print("=" * 60)

    config = PipelineConfig(
        model_name="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        enable_bib_verification=True,
        crossref_mailto="your-email@example.com",
    )

    pipeline = PaperReviewPipeline(config)

    print(
        """
Usage:
    result = await pipeline.review_tex_package("paper_source.tar.gz")

    print(f"TeX Info:")
    print(f"  Main file: {result.tex_info.main_tex}")
    print(f"  Total files: {result.tex_info.total_files}")

    if result.bib_verification:
        for bib_file, summary in result.bib_verification.items():
            print(f"\\n{bib_file}:")
            print(f"  Verified: {summary.verified}/{summary.total_references}")
            print(f"  Suspect: {summary.suspect}")
            if summary.suspect_references:
                print(f"  Suspect refs: {summary.suspect_references}")
"""
    )


def main():
    demo_tex_processor()
    asyncio.run(demo_full_pipeline())


if __name__ == "__main__":
    main()
