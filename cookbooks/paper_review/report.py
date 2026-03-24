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

    # Rebuttal Draft
    if result.rebuttal:
        lines.extend(
            [
                "---",
                "",
                "## Rebuttal Draft",
                "",
            ]
        )
        if result.rebuttal.concerns:
            lines.append("### Concerns Identified")
            lines.append("")
            lines.append("| # | Severity | Type | Concern |")
            lines.append("|---|----------|------|---------|")
            for i, c in enumerate(result.rebuttal.concerns, 1):
                sev_icon = "\U0001f534" if c.severity == "major" else "\U0001f7e1"
                type_label = (
                    "\U0001f527 Action Required" if c.response_type == "action_required" else "\U0001f4ac Clarification"
                )
                lines.append(f"| {i} | {sev_icon} {c.severity.title()} | {type_label} | {c.concern} |")
            lines.append("")

            lines.append("### Point-by-Point Responses")
            lines.append("")
            for i, c in enumerate(result.rebuttal.concerns, 1):
                lines.append(f"**Concern {i}** ({c.severity}): {c.concern}")
                lines.append("")
                lines.append(f"> {c.draft_response}")
                lines.append("")
        if result.rebuttal.rebuttal_text:
            lines.append("### Full Rebuttal Text")
            lines.append("")
            lines.append(result.rebuttal.rebuttal_text)
            lines.append("")
        if result.rebuttal.general_suggestions:
            lines.append("### Suggestions for Revision")
            lines.append("")
            for s in result.rebuttal.general_suggestions:
                lines.append(f"- {s}")
            lines.append("")

    # Rebuttal Assessment
    if result.rebuttal_assessment:
        ra = result.rebuttal_assessment
        score_delta = ra.updated_score - ra.original_score
        if score_delta > 0:
            delta_str = f" (+{score_delta})"
        elif score_delta < 0:
            delta_str = f" ({score_delta})"
        else:
            delta_str = " (unchanged)"

        lines.extend(
            [
                "---",
                "",
                "## Rebuttal Assessment",
                "",
                f"**Original Score**: {ra.original_score}/6 \u2192 **Updated Score**: {ra.updated_score}/6{delta_str}",
                "",
                _score_bar(ra.updated_score, 6),
                "",
            ]
        )
        if ra.score_change_reasoning:
            lines.extend(["**Score Change Reasoning**: " + ra.score_change_reasoning, ""])
        if ra.overall_assessment:
            lines.extend(["### Overall Assessment", "", ra.overall_assessment, ""])

        if ra.point_assessments:
            lines.append("### Point-by-Point Assessment")
            lines.append("")
            lines.append("| # | Concern | Adequacy | Reasoning |")
            lines.append("|---|---------|----------|-----------|")
            adequacy_icons = {
                "fully_addressed": "\u2705 Fully Addressed",
                "partially_addressed": "\u26a0\ufe0f Partially Addressed",
                "not_addressed": "\u274c Not Addressed",
            }
            for i, p in enumerate(ra.point_assessments, 1):
                icon = adequacy_icons.get(p.adequacy, p.adequacy)
                lines.append(f"| {i} | {p.concern} | {icon} | {p.reasoning} |")
            lines.append("")

        if ra.rebuttal_strengths:
            lines.append("### Rebuttal Strengths")
            lines.append("")
            for s in ra.rebuttal_strengths:
                lines.append(f"- {s}")
            lines.append("")

        if ra.unresolved_concerns:
            lines.append("### Unresolved Concerns")
            lines.append("")
            for c in ra.unresolved_concerns:
                lines.append(f"- {c}")
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
