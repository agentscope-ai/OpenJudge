#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review a LaTeX source package or verify a standalone BibTeX file.

Usage:
    python review_tex.py paper_source.tar.gz
    python review_tex.py paper_source.zip --discipline biology
    python review_tex.py --bib-only references.bib --email you@example.com
    python review_tex.py paper_source.tar.gz --language zh
    python review_tex.py paper_source.tar.gz --instructions "Focus on methodology"
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
    """Parse CLI arguments for TeX package or BibTeX review."""
    parser = argparse.ArgumentParser(
        description="Review a LaTeX source package or BibTeX file with OpenJudge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python review_tex.py paper_source.tar.gz
  python review_tex.py paper_source.zip --discipline biology --venue "Cell 2025"
  python review_tex.py --bib-only references.bib --email you@example.com
  python review_tex.py paper_source.tar.gz --model claude-opus-4-5
        """,
    )
    # Source input: either a TeX package or --bib-only mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("package", nargs="?", metavar="PACKAGE", help="Path to .tar.gz or .zip TeX source package")
    group.add_argument("--bib-only", metavar="BIB_FILE", help="Verify a standalone .bib file (no PDF review)")

    parser.add_argument("--model", default="gpt-4o", metavar="MODEL", help="Model name (default: gpt-4o)")
    parser.add_argument("--api-key", metavar="KEY", help="API key (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--base-url", metavar="URL", help="Custom API base URL")
    parser.add_argument(
        "--discipline",
        choices=DISCIPLINES,
        metavar="DISCIPLINE",
        help=f"Academic discipline. Options: {', '.join(DISCIPLINES)}",
    )
    parser.add_argument("--venue", metavar="VENUE", help="Target venue, e.g. 'NeurIPS 2025'")
    parser.add_argument("--paper-name", metavar="NAME", help="Paper title for the report")
    parser.add_argument("--output", metavar="FILE", help="Output .md report path")
    parser.add_argument("--email", metavar="EMAIL", help="Your email for CrossRef API (recommended for BibTeX check)")
    parser.add_argument("--no-safety", action="store_true", help="Skip safety checks")
    parser.add_argument("--no-correctness", action="store_true", help="Skip correctness check")
    parser.add_argument("--no-criticality", action="store_true", help="Skip criticality verification")
    parser.add_argument("--no-bib", action="store_true", help="Skip BibTeX verification")
    parser.add_argument(
        "--language",
        metavar="LANG",
        choices=["en", "zh"],
        help="Output language for the review: 'en' (default) or 'zh' (Simplified Chinese)",
    )
    parser.add_argument(
        "--instructions", metavar="TEXT", help="Free-form reviewer instructions, e.g. 'Focus on experimental design'"
    )
    parser.add_argument("--timeout", type=int, default=7500, metavar="SECONDS", help="API timeout in seconds")
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


def print_bib_results(bib_verification: dict):
    """Print BibTeX verification results."""
    for bib_file, summary in bib_verification.items():
        rate = summary.verification_rate
        print(f"\nBibTeX: {Path(bib_file).name}")
        print(f"  Total references : {summary.total_references}")
        print(f"  Verified         : {summary.verified} ({rate:.0%})")
        print(f"  Suspect          : {summary.suspect}")
        if summary.errors:
            print(f"  Errors           : {summary.errors}")
        if summary.suspect_references:
            print("\n  ⚠ Suspect references (check manually):")
            for ref in summary.suspect_references:
                print(f"    - {ref}")


async def run_bib_only(bib_path: str, email: str | None):
    """Verify a standalone .bib file without running a full paper review."""
    from cookbooks.paper_review.processors.bib_checker import (
        BibChecker,
        VerificationStatus,
    )

    bib_file = Path(bib_path)
    if not bib_file.exists():
        print(f"ERROR: File not found: {bib_file}")
        sys.exit(1)

    print(f"Verifying BibTeX: {bib_file.name}")
    print()

    checker = BibChecker(mailto=email)
    results = checker.check_bib_file(str(bib_file))

    for r in results["results"]:
        icon = "✓" if r.status == VerificationStatus.VERIFIED else "✗"
        print(f"  {icon} [{r.status.value.upper():8s}] {r.reference.title}")
        if r.status != VerificationStatus.VERIFIED:
            print(f"           {r.message}")

    print("\n" + "=" * 50)
    verified = results["verified"]
    total = results["total_references"]
    rate = results["verification_rate"]
    print(f"Verified : {verified}/{total} ({rate:.0%})")
    print(f"Suspect  : {results['suspect']}")
    if results["suspect"] > 0:
        print("⚠ Run a full review to get suspect references listed in the report.")


async def run_tex_review(args):
    """Run review pipeline for a TeX source package."""
    from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig
    from cookbooks.paper_review.report import generate_report

    package_path = Path(args.package)
    if not package_path.exists():
        print(f"ERROR: File not found: {package_path}")
        sys.exit(1)

    if package_path.suffix not in (".gz", ".zip") and ".tar" not in package_path.name:
        print(f"WARNING: Expected .tar.gz or .zip, got: {package_path.suffix}")

    paper_name = args.paper_name or package_path.stem.replace(".tar", "")
    output_path = args.output or f"{paper_name}_review.md"
    api_key = resolve_api_key(args.api_key, args.model)

    if not api_key:
        print("WARNING: No API key found. Set OPENAI_API_KEY or pass --api-key.")

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
        enable_bib_verification=not args.no_bib,
        crossref_mailto=args.email,
    )

    pipeline = PaperReviewPipeline(config)

    print(f"Processing TeX package: {package_path.name}")
    if args.discipline:
        print(f"Discipline: {args.discipline}")
    if args.venue:
        print(f"Venue: {args.venue}")
    print(f"Model: {args.model}")
    print()

    # Step 1: Parse TeX package (extracts bib, figures, merges .tex files)
    result = await pipeline.review_tex_package(str(package_path), package_name=paper_name)

    if result.tex_info:
        print("TeX package parsed:")
        print(f"  Main file   : {result.tex_info.main_tex}")
        print(f"  Total files : {result.tex_info.total_files}")
        print(f"  BibTeX files: {result.tex_info.bib_files}")
        print(f"  Figures     : {len(result.tex_info.figures)}")
        print()

    if result.bib_verification:
        print_bib_results(result.bib_verification)

    # Generate report
    report = generate_report(result, paper_name=paper_name, output_path=output_path)
    print("\n" + report)
    print(f"\n✓ Report saved: {output_path}")


def main():
    """Entry point for the review_tex script."""
    check_install()
    args = parse_args()

    if args.bib_only:
        asyncio.run(run_bib_only(args.bib_only, args.email))
    else:
        asyncio.run(run_tex_review(args))


if __name__ == "__main__":
    main()
