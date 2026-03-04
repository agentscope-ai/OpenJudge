#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review a single PDF paper using the OpenJudge PaperReviewPipeline.

Usage:
    python review_paper.py paper.pdf
    python review_paper.py paper.pdf --discipline cs --venue "NeurIPS 2025"
    python review_paper.py paper.pdf --bib refs.bib --email you@example.com
    python review_paper.py paper.pdf --model claude-opus-4-5
    python review_paper.py paper.pdf --language zh
    python review_paper.py paper.pdf --instructions "Focus on experimental design"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

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


def parse_args():
    """Parse CLI arguments for PDF paper review."""
    parser = argparse.ArgumentParser(
        description="Review an academic paper PDF with OpenJudge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python review_paper.py paper.pdf
  python review_paper.py paper.pdf --discipline cs --venue "NeurIPS 2025"
  python review_paper.py paper.pdf --bib references.bib --email you@example.com
  python review_paper.py paper.pdf --model claude-opus-4-5 --api-key $ANTHROPIC_API_KEY
  python review_paper.py paper.pdf --base-url http://localhost:11434/v1 --model ollama/llama3
        """,
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--bib", metavar="BIB_FILE", help="Path to .bib file for reference verification")
    parser.add_argument("--model", default="gpt-4o", metavar="MODEL", help="Model name (default: gpt-4o)")
    parser.add_argument(
        "--api-key", metavar="KEY", help="API key (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY env vars)"
    )
    parser.add_argument("--base-url", metavar="URL", help="Custom API base URL for self-hosted or proxy endpoints")
    parser.add_argument(
        "--discipline",
        choices=DISCIPLINES,
        metavar="DISCIPLINE",
        help=f"Academic discipline. Options: {', '.join(DISCIPLINES)}",
    )
    parser.add_argument("--venue", metavar="VENUE", help="Target venue, e.g. 'NeurIPS 2025' or 'The Lancet'")
    parser.add_argument("--paper-name", metavar="NAME", help="Paper title for the report (default: PDF filename)")
    parser.add_argument("--output", metavar="FILE", help="Output .md report path (default: <paper_name>_review.md)")
    parser.add_argument(
        "--email", metavar="EMAIL", help="Your email for CrossRef API (better rate limits in BibTeX check)"
    )
    parser.add_argument("--no-safety", action="store_true", help="Skip safety checks")
    parser.add_argument("--no-correctness", action="store_true", help="Skip correctness check")
    parser.add_argument("--no-criticality", action="store_true", help="Skip criticality verification")
    parser.add_argument("--no-bib", action="store_true", help="Skip BibTeX verification even if --bib is given")
    parser.add_argument(
        "--vision", action="store_true", help="Use vision mode (render PDF pages as images). Requires pypdfium2."
    )
    parser.add_argument(
        "--vision-max-pages",
        type=int,
        default=30,
        metavar="N",
        help="Max pages to send in vision mode (default: 30). Set 0 for all pages.",
    )
    parser.add_argument(
        "--format-vision-max-pages",
        type=int,
        default=10,
        metavar="N",
        help="Max pages for format check in vision mode (default: 10). Set 0 to use --vision-max-pages.",
    )
    parser.add_argument(
        "--language",
        metavar="LANG",
        choices=["en", "zh"],
        help="Output language for the review: 'en' (default) or 'zh' (Simplified Chinese)",
    )
    parser.add_argument(
        "--instructions", metavar="TEXT", help="Free-form reviewer instructions, e.g. 'Focus on experimental design'"
    )
    parser.add_argument(
        "--timeout", type=int, default=7500, metavar="SECONDS", help="API timeout in seconds (default: 7500)"
    )
    return parser.parse_args()


def resolve_api_key(args_key: str | None, model: str) -> str:
    """Resolve API key from CLI arg or environment variables."""
    if args_key:
        return args_key
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


def check_install():
    """Ensure paper review dependencies are importable."""
    try:
        __import__("cookbooks.paper_review")

        return True
    except ImportError:
        print("ERROR: OpenJudge paper_review not found.\n")
        print("Install it first:")
        print("  git clone https://github.com/agentscope-ai/OpenJudge.git")
        print("  cd OpenJudge && pip install -e .")
        print("  pip install litellm httpx")
        sys.exit(1)


async def run_review(args):
    """Run review pipeline for a single PDF file."""
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    paper_name = args.paper_name or pdf_path.stem
    output_path = args.output or f"{paper_name}_review.md"
    api_key = resolve_api_key(args.api_key, args.model)
    has_bib = bool(args.bib) and not args.no_bib

    if not api_key:
        print("WARNING: No API key found. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY) or pass --api-key.")

    config = PipelineConfig(
        model_name=args.model,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        discipline=args.discipline,
        venue=args.venue,
        instructions=args.instructions,
        language=args.language,
        enable_safety_checks=not args.no_safety,
        enable_correctness=not args.no_correctness,
        enable_review=True,
        enable_criticality=not args.no_criticality,
        enable_bib_verification=has_bib,
        crossref_mailto=args.email,
        use_vision_for_pdf=args.vision,
        vision_max_pages=args.vision_max_pages or None,
        format_vision_max_pages=args.format_vision_max_pages or None,
    )

    pipeline = PaperReviewPipeline(config)

    print(f"Reviewing: {pdf_path.name}")
    if args.discipline:
        print(f"Discipline: {args.discipline}")
    if args.venue:
        print(f"Venue: {args.venue}")
    if has_bib:
        print(f"BibTeX: {args.bib}")
    print(f"Model: {args.model}")
    print()

    result, report = await pipeline.review_and_report(
        pdf_input=str(pdf_path),
        paper_name=paper_name,
        bib_path=args.bib if has_bib else None,
        output_path=output_path,
    )

    print(report)
    print(f"\n✓ Report saved: {output_path}")

    # Print quick summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if result.correctness:
        print(f"Correctness: {result.correctness.score}/3")
        if result.correctness.key_issues:
            print(f"  Issues: {len(result.correctness.key_issues)} found")
    if result.review:
        print(f"Review score: {result.review.score}/6")
    if result.criticality:
        print(f"Criticality: {result.criticality.score}/3")
    if result.bib_verification:
        for bib_file, summary in result.bib_verification.items():
            rate = summary.verification_rate
            print(
                f"BibTeX ({Path(bib_file).name}): {summary.verified}/{summary.total_references} verified ({rate:.0%})"
            )
            if summary.suspect > 0:
                print(f"  ⚠ {summary.suspect} suspect references — check manually")


def main():
    """Entry point for the review_paper script."""
    check_install()
    args = parse_args()
    asyncio.run(run_review(args))


if __name__ == "__main__":
    main()
