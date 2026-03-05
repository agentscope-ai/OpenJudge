# -*- coding: utf-8 -*-
"""CLI entry point for Paper Review.

Usage:
    # Review a PDF
    python -m cookbooks.paper_review paper.pdf
    python -m cookbooks.paper_review paper.pdf --discipline cs --venue "NeurIPS 2025"
    python -m cookbooks.paper_review paper.pdf --bib refs.bib --email you@example.com
    python -m cookbooks.paper_review paper.pdf --language zh
    python -m cookbooks.paper_review paper.pdf --vision

    # Review a TeX source package
    python -m cookbooks.paper_review paper_source.tar.gz
    python -m cookbooks.paper_review paper_source.zip --discipline biology

    # Verify a standalone BibTeX file
    python -m cookbooks.paper_review --bib-only references.bib --email you@example.com
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import fire
from loguru import logger

DISCIPLINES = [
    "cs",
    "medicine",
    "physics",
    "chemistry",
    "biology",
    "economics",
    "psychology",
    "environmental_science",
    "mathematics",
    "social_sciences",
]


def _resolve_api_key(api_key: Optional[str], model: str) -> str:
    if api_key:
        return api_key
    if model.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if model.startswith("qwen") or model.startswith("dashscope"):
        return os.environ.get("DASHSCOPE_API_KEY", "")
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    )


def _detect_mode(input_path: str) -> str:
    """Detect input file type: 'pdf', 'tex', or 'bib'."""
    p = Path(input_path)
    suffix = p.suffix.lower()
    name = p.name.lower()

    if suffix == ".pdf":
        return "pdf"
    if suffix in (".zip",) or ".tar" in name:
        return "tex"
    if suffix == ".bib":
        return "bib"

    logger.warning(f"Cannot auto-detect file type for '{p.name}', treating as PDF")
    return "pdf"


async def _run_bib_only(bib_path: str, email: Optional[str]) -> None:
    from cookbooks.paper_review.processors.bib_checker import (
        BibChecker,
        VerificationStatus,
    )

    bib_file = Path(bib_path)
    if not bib_file.exists():
        logger.error(f"File not found: {bib_file}")
        sys.exit(1)

    logger.info(f"Verifying BibTeX: {bib_file.name}")
    checker = BibChecker(mailto=email)
    results = checker.check_bib_file(str(bib_file))

    for r in results["results"]:
        icon = "✓" if r.status == VerificationStatus.VERIFIED else "✗"
        print(f"  {icon} [{r.status.value.upper():8s}] {r.reference.title}")
        if r.status != VerificationStatus.VERIFIED:
            print(f"           {r.message}")

    print("\n" + "=" * 50)
    print(f"Verified : {results['verified']}/{results['total_references']} ({results['verification_rate']:.0%})")
    print(f"Suspect  : {results['suspect']}")


async def _run_pdf_review(
    pdf_path: str,
    *,
    model: str,
    api_key: str,
    base_url: Optional[str],
    discipline: Optional[str],
    venue: Optional[str],
    instructions: Optional[str],
    language: Optional[str],
    paper_name: Optional[str],
    output: Optional[str],
    bib: Optional[str],
    email: Optional[str],
    no_safety: bool,
    no_correctness: bool,
    no_criticality: bool,
    no_bib: bool,
    vision: bool,
    vision_max_pages: int,
    format_vision_max_pages: int,
    timeout: int,
) -> None:
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig

    pdf = Path(pdf_path)
    if not pdf.exists():
        logger.error(f"File not found: {pdf}")
        sys.exit(1)

    name = paper_name or pdf.stem
    out = output or f"{name}_review.md"
    has_bib = bool(bib) and not no_bib

    config = PipelineConfig(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        discipline=discipline,
        venue=venue,
        instructions=instructions,
        language=language,
        enable_safety_checks=not no_safety,
        enable_correctness=not no_correctness,
        enable_review=True,
        enable_criticality=not no_criticality,
        enable_bib_verification=has_bib,
        crossref_mailto=email,
        use_vision_for_pdf=vision,
        vision_max_pages=vision_max_pages or None,
        format_vision_max_pages=format_vision_max_pages or None,
    )

    pipeline = PaperReviewPipeline(config)

    logger.info(f"Reviewing: {pdf.name}  Model: {model}")
    if discipline:
        logger.info(f"Discipline: {discipline}")
    if venue:
        logger.info(f"Venue: {venue}")

    result, report = await pipeline.review_and_report(
        pdf_input=str(pdf),
        paper_name=name,
        bib_path=bib if has_bib else None,
        output_path=out,
    )

    print(report)
    logger.info(f"Report saved: {out}")

    if result.correctness:
        logger.info(f"Correctness: {result.correctness.score}/3")
    if result.review:
        logger.info(f"Review score: {result.review.score}/6")
    if result.criticality:
        logger.info(f"Criticality: {result.criticality.score}/3")
    if result.bib_verification:
        for bib_file, summary in result.bib_verification.items():
            logger.info(
                f"BibTeX ({Path(bib_file).name}): "
                f"{summary.verified}/{summary.total_references} verified ({summary.verification_rate:.0%})"
            )


async def _run_tex_review(
    package_path: str,
    *,
    model: str,
    api_key: str,
    base_url: Optional[str],
    discipline: Optional[str],
    venue: Optional[str],
    instructions: Optional[str],
    language: Optional[str],
    paper_name: Optional[str],
    output: Optional[str],
    email: Optional[str],
    no_safety: bool,
    no_correctness: bool,
    no_criticality: bool,
    no_bib: bool,
    timeout: int,
) -> None:
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig
    from cookbooks.paper_review.report import generate_report

    pkg = Path(package_path)
    if not pkg.exists():
        logger.error(f"File not found: {pkg}")
        sys.exit(1)

    name = paper_name or pkg.stem.replace(".tar", "")
    out = output or f"{name}_review.md"

    config = PipelineConfig(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        discipline=discipline,
        venue=venue,
        instructions=instructions,
        language=language,
        enable_safety_checks=not no_safety,
        enable_correctness=not no_correctness,
        enable_review=True,
        enable_criticality=not no_criticality,
        enable_bib_verification=not no_bib,
        crossref_mailto=email,
    )

    pipeline = PaperReviewPipeline(config)

    logger.info(f"Processing TeX package: {pkg.name}  Model: {model}")
    if discipline:
        logger.info(f"Discipline: {discipline}")

    result = await pipeline.review_tex_package(str(pkg), package_name=name)

    if result.tex_info:
        logger.info(
            f"TeX parsed: main={result.tex_info.main_tex}, "
            f"files={result.tex_info.total_files}, "
            f"bib={result.tex_info.bib_files}, "
            f"figures={len(result.tex_info.figures)}"
        )

    report = generate_report(result, paper_name=name, output_path=out)
    print(report)
    logger.info(f"Report saved: {out}")


def main(
    input: Optional[str] = None,
    bib_only: Optional[str] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    discipline: Optional[str] = None,
    venue: Optional[str] = None,
    instructions: Optional[str] = None,
    language: Optional[str] = None,
    paper_name: Optional[str] = None,
    output: Optional[str] = None,
    bib: Optional[str] = None,
    email: Optional[str] = None,
    no_safety: bool = False,
    no_correctness: bool = False,
    no_criticality: bool = False,
    no_bib: bool = False,
    vision: bool = True,
    vision_max_pages: int = 30,
    format_vision_max_pages: int = 10,
    timeout: int = 7500,
) -> None:
    """Paper Review CLI — review PDFs, TeX packages, or verify BibTeX files.

    File type is auto-detected: .pdf → PDF review, .tar.gz/.zip → TeX review,
    .bib → BibTeX verification. Use --bib_only for explicit BibTeX-only mode.

    Args:
        input: Path to PDF, TeX package (.tar.gz/.zip), or .bib file.
        bib_only: Path to a .bib file for standalone verification (no paper review).
        model: Model name (default: gpt-4o).
        api_key: API key (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY env vars).
        base_url: Custom API base URL.
        discipline: Academic discipline (cs, medicine, physics, chemistry, biology,
                    economics, psychology, environmental_science, mathematics, social_sciences).
        venue: Target conference/journal, e.g. "NeurIPS 2025".
        instructions: Free-form reviewer guidance.
        language: Output language: "en" (default) or "zh".
        paper_name: Paper title in report (default: filename stem).
        output: Output .md report path.
        bib: Path to .bib file for reference verification alongside PDF review.
        email: CrossRef mailto for better rate limits.
        no_safety: Skip safety checks.
        no_correctness: Skip correctness check.
        no_criticality: Skip criticality verification.
        no_bib: Skip BibTeX verification.
        vision: Use vision mode for PDF (default: True, requires pypdfium2).
        vision_max_pages: Max pages in vision mode (default: 30, 0 = all).
        format_vision_max_pages: Max pages for format check in vision mode (default: 10).
        timeout: API timeout in seconds (default: 7500).

    Examples:
        python -m cookbooks.paper_review paper.pdf
        python -m cookbooks.paper_review paper.pdf --discipline cs --venue "NeurIPS 2025"
        python -m cookbooks.paper_review paper.pdf --bib refs.bib --email you@example.com
        python -m cookbooks.paper_review paper.pdf --vision --language zh
        python -m cookbooks.paper_review paper_source.tar.gz --discipline biology
        python -m cookbooks.paper_review --bib_only references.bib --email you@example.com
    """
    if bib_only:
        asyncio.run(_run_bib_only(bib_only, email))
        return

    if not input:
        logger.error("Please provide an input file path, or use --bib_only for standalone BibTeX verification.")
        logger.error("Usage: python -m cookbooks.paper_review <file> [options]")
        sys.exit(1)

    mode = _detect_mode(input)
    resolved_key = _resolve_api_key(api_key, model)

    if not resolved_key:
        logger.warning("No API key found. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY) or pass --api_key.")

    if mode == "bib":
        asyncio.run(_run_bib_only(input, email))
    elif mode == "tex":
        asyncio.run(
            _run_tex_review(
                input,
                model=model,
                api_key=resolved_key,
                base_url=base_url,
                discipline=discipline,
                venue=venue,
                instructions=instructions,
                language=language,
                paper_name=paper_name,
                output=output,
                email=email,
                no_safety=no_safety,
                no_correctness=no_correctness,
                no_criticality=no_criticality,
                no_bib=no_bib,
                timeout=timeout,
            )
        )
    else:
        asyncio.run(
            _run_pdf_review(
                input,
                model=model,
                api_key=resolved_key,
                base_url=base_url,
                discipline=discipline,
                venue=venue,
                instructions=instructions,
                language=language,
                paper_name=paper_name,
                output=output,
                bib=bib,
                email=email,
                no_safety=no_safety,
                no_correctness=no_correctness,
                no_criticality=no_criticality,
                no_bib=no_bib,
                vision=vision,
                vision_max_pages=vision_max_pages,
                format_vision_max_pages=format_vision_max_pages,
                timeout=timeout,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
