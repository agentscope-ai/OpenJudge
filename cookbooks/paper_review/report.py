# -*- coding: utf-8 -*-
"""Generate Markdown reports from paper review results."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from cookbooks.paper_review.schema import PaperReviewResult


def generate_report(
    result: PaperReviewResult,
    paper_name: str = "Paper",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a Markdown report from review results.

    Args:
        result: PaperReviewResult from pipeline
        paper_name: Name of the reviewed paper
        output_path: Optional path to save the report

    Returns:
        Markdown formatted report string
    """
    lines = [
        f"# Paper Review Report: {paper_name}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Safety Status
    lines.extend(
        [
            "## 1. Safety Check",
            "",
            f"**Status**: {'✅ Safe' if result.is_safe else '❌ Issues Detected'}",
            "",
        ]
    )
    if result.safety_issues:
        lines.append("**Issues**:")
        for issue in result.safety_issues:
            lines.append(f"- {issue}")
        lines.append("")

    if result.format_compliant is not None:
        lines.append(f"**Format Compliant**: {'✅ Yes' if result.format_compliant else '⚠️ No'}")
        lines.append("")

    # Review Score
    if result.review:
        lines.extend(
            [
                "---",
                "",
                "## 2. Paper Review",
                "",
                f"**Score**: {result.review.score}/6",
                "",
                _score_bar(result.review.score, 6),
                "",
                "### Review Comments",
                "",
                result.review.review,
                "",
            ]
        )

    # Correctness Check
    if result.correctness:
        # Convert to positive scoring: 1->3 (best), 2->2, 3->1 (worst)
        display_score = 4 - result.correctness.score
        score_labels = {
            3: "No objective errors detected",
            2: "Minor errors present",
            1: "Major errors detected",
        }
        lines.extend(
            [
                "---",
                "",
                "## 3. Correctness Analysis",
                "",
                f"**Score**: {display_score}/3 - {score_labels.get(display_score, '')}",
                "",
                _score_bar(display_score, 3),
                "",
                "### Reasoning",
                "",
                result.correctness.reasoning,
                "",
            ]
        )
        if result.correctness.key_issues:
            lines.extend(
                [
                    "### Key Issues",
                    "",
                ]
            )
            for i, issue in enumerate(result.correctness.key_issues, 1):
                lines.append(f"{i}. {issue}")
            lines.append("")

    # Criticality Verification
    if result.criticality:
        # Convert to positive scoring: 1->3 (best), 2->2, 3->1 (worst)
        display_score = 4 - result.criticality.score
        score_labels = {
            3: "No genuine errors (false positives)",
            2: "Minor errors, main contributions valid",
            1: "Major errors compromising validity",
        }
        lines.extend(
            [
                "---",
                "",
                "## 4. Criticality Verification",
                "",
                f"**Score**: {display_score}/3 - {score_labels.get(display_score, '')}",
                "",
                _score_bar(display_score, 3),
                "",
                "### Reasoning",
                "",
                result.criticality.reasoning,
                "",
            ]
        )
        issues = result.criticality.issues
        if issues:
            if issues.major:
                lines.append("### Major Issues")
                lines.append("")
                for issue in issues.major:
                    lines.append(f"- 🔴 {issue}")
                lines.append("")
            if issues.minor:
                lines.append("### Minor Issues")
                lines.append("")
                for issue in issues.minor:
                    lines.append(f"- 🟡 {issue}")
                lines.append("")
            if issues.false_positives:
                lines.append("### False Positives")
                lines.append("")
                for issue in issues.false_positives:
                    lines.append(f"- ⚪ {issue}")
                lines.append("")

    # BibTeX Verification
    if result.bib_verification:
        lines.extend(
            [
                "---",
                "",
                "## 5. Reference Verification",
                "",
            ]
        )
        for bib_file, summary in result.bib_verification.items():
            lines.extend(
                [
                    f"### {bib_file}",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Total References | {summary.total_references} |",
                    f"| Verified | {summary.verified} |",
                    f"| Suspect | {summary.suspect} |",
                    f"| Errors | {summary.errors} |",
                    f"| Verification Rate | {summary.verification_rate:.1%} |",
                    "",
                ]
            )
            if summary.suspect_references:
                lines.append("**Suspect References**:")
                for ref in summary.suspect_references[:10]:  # Limit to 10
                    lines.append(f"- {ref}")
                if len(summary.suspect_references) > 10:
                    lines.append(f"- ... and {len(summary.suspect_references) - 10} more")
                lines.append("")

    # TeX Package Info
    if result.tex_info:
        lines.extend(
            [
                "---",
                "",
                "## 6. TeX Package Info",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| Main TeX File | `{result.tex_info.main_tex}` |",
                f"| Total TeX Files | {result.tex_info.total_files} |",
                f"| BibTeX Files | {len(result.tex_info.bib_files)} |",
                f"| Figures | {len(result.tex_info.figures)} |",
                "",
            ]
        )

    # Footer
    lines.extend(
        [
            "---",
            "",
            "*Generated by OpenJudge Paper Review Cookbook*",
        ]
    )

    report = "\n".join(lines)

    # Save if path provided
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def _score_bar(score: int, max_score: int) -> str:
    """Generate a visual score bar (higher is always better)."""
    filled = score
    empty = max_score - score
    return f"{'🟢' * filled}{'⚪' * empty} ({score}/{max_score})"
