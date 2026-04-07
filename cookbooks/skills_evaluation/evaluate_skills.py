# -*- coding: utf-8 -*-
"""
Skills Evaluation Example

Runs SkillsGradingRunner on a skills directory and prints the results.

Usage:
    python cookbooks/skills_evaluation/evaluate_skills.py [SKILLS_DIR] [TASK_DESCRIPTION]

Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL from the .env file
(or environment variables) automatically.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from project root without installing the package
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from cookbooks.skills_evaluation.runner import (  # noqa: E402
    SkillGradingResult,
    SkillsGradingRunner,
    build_markdown_report,
)
from openjudge.models.openai_chat_model import OpenAIChatModel  # noqa: E402


def _build_model() -> OpenAIChatModel:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("OPENAI_MODEL", "qwen3.6-plus")
    return OpenAIChatModel(model=model_name, api_key=api_key, base_url=base_url)


def _print_result(result: SkillGradingResult) -> None:
    verdict = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n{'=' * 60}")
    print(f"Skill : {result.skill_name}")
    print(f"Path  : {result.skill_path}")
    print(f"Score : {result.weighted_score:.3f}  {verdict}")
    print(f"Time  : {result.grading_duration_seconds:.1f}s")
    print(f"{'─' * 60}")
    for dim_name, dim in result.dimension_scores.items():
        status = "✅" if dim.passed else "❌"
        if dim.error:
            print(f"  [{dim_name:<17}] ERROR  — {dim.error}")
        else:
            print(
                f"  [{dim_name:<17}] {status}  score={dim.score:.0f}"
                f"  norm={dim.normalized_score:.2f}"
                f"  w={dim.weight:.1f}"
            )
            if dim.reason:
                reason_preview = dim.reason[:120].replace("\n", " ")
                print(f"    reason: {reason_preview}{'…' if len(dim.reason) > 120 else ''}")
    if result.errors:
        print(f"  Errors: {result.errors}")


async def main(skills_dir: str, task_description: str | None = None) -> None:
    model = _build_model()
    runner = SkillsGradingRunner(
        model=model,
        weights={
            "threat_analysis": 1.0,
            "alignment": 1.0,
            "completeness": 1.0,
            "relevance": 1.0,
            "structure": 1.0,
        },
    )

    print(f"Evaluating skills in: {skills_dir}")
    if task_description:
        print(f"Task description: {task_description}")
    results = await runner.arun(skills_dir, task_description=task_description)

    for r in results:
        _print_result(r)

    print(f"\n{'=' * 60}")
    print(f"Total skills evaluated: {len(results)}")
    passed = sum(1 for r in results if r.passed)
    print(f"Passed: {passed} / {len(results)}")

    out_dir = _ROOT / "cookbooks" / "skills_evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = out_dir / "grading_results.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON  → {json_path}")

    # Markdown report
    md_path = out_dir / "grading_report.md"
    md_path.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"MD    → {md_path}")

    # Print markdown to stdout as well
    print()
    print(build_markdown_report(results))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / ".agents" / "skills" / "financial-consulting-research")
    task_desc = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(target, task_description=task_desc))
