# -*- coding: utf-8 -*-
"""Result panel component for Paper Review feature.

Displays comprehensive review results with scores, issues, and downloadable report.
"""

import html
from typing import Optional

import streamlit as st
from features.paper_review.services.pipeline_runner import ReviewTaskResult
from shared.i18n import t

from cookbooks.paper_review import PaperReviewResult


def _get_score_color(score: int, max_score: int) -> str:
    """Get color based on score percentage."""
    pct = score / max_score
    if pct >= 0.7:
        return "#22C55E"  # Green
    elif pct >= 0.4:
        return "#F59E0B"  # Yellow
    else:
        return "#EF4444"  # Red


def _render_score_card(
    title: str,
    score: int,
    max_score: int,
    icon: str,
    description: Optional[str] = None,
) -> None:
    """Render a score card with visual indicator."""
    color = _get_score_color(score, max_score)
    pct = (score / max_score) * 100

    # Build description HTML separately to avoid f-string issues
    desc_html = ""
    if description:
        desc_text = f"{description[:200]}..." if len(description) > 200 else description
        desc_html = (
            '<div style="color: #94A3B8; font-size: 0.85rem; '
            f'margin-top: 0.75rem; line-height: 1.5;">{desc_text}</div>'
        )

    # fmt: off
    html_content = f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span style="font-weight: 600; color: #F1F5F9;">{title}</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 2rem; font-weight: 700; color: {color};">{score}</span>
            <span style="color: #64748B;">/ {max_score}</span>
        </div>
        <div style="height: 6px; background: #1E293B; border-radius: 3px; overflow: hidden;">
            <div style="width: {pct}%; height: 100%; background: {color}; border-radius: 3px;"></div>
        </div>
        {desc_html}
    </div>
    """
    # fmt: on
    st.markdown(html_content, unsafe_allow_html=True)


def _render_safety_result(result: PaperReviewResult) -> None:
    """Render safety check results."""
    is_safe = result.is_safe
    icon = "✅" if is_safe else "❌"
    status = t("paper_review.result.safety_passed") if is_safe else t("paper_review.result.safety_failed")
    color = "#22C55E" if is_safe else "#EF4444"
    bg = "rgba(34, 197, 94, 0.1)" if is_safe else "rgba(239, 68, 68, 0.1)"
    border = "rgba(34, 197, 94, 0.3)" if is_safe else "rgba(239, 68, 68, 0.3)"

    st.markdown(
        f"""<div style="
            background: {bg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">🛡️</span>
                <span style="font-weight: 600; color: #F1F5F9;">{t("paper_review.result.safety")}</span>
                <span style="margin-left: auto; color: {color}; font-weight: 600;">{icon} {status}</span>
            </div>
            {f'<div style="color: #94A3B8; font-size: 0.85rem;">'
             f'{t("paper_review.result.format_compliant")}: '
             f'{"✓" if result.format_compliant else "✗"}</div>'
             if result.format_compliant is not None else ''}
        </div>""",
        unsafe_allow_html=True,
    )

    # Show safety issues if any
    if result.safety_issues:
        for issue in result.safety_issues:
            st.warning(issue)


def _render_review_result(result: PaperReviewResult) -> None:
    """Render paper review results."""
    if not result.review:
        return

    _render_score_card(
        title=t("paper_review.result.review_score"),
        score=result.review.score,
        max_score=6,
        icon="📝",
    )

    # Show full review in expander
    with st.expander(t("paper_review.help.stage_review"), expanded=False):
        st.markdown(result.review.review)


def _render_correctness_result(result: PaperReviewResult) -> None:
    """Render correctness analysis results."""
    if not result.correctness:
        return

    # Score mapping: 1 = no errors (best), 3 = major errors (worst)
    # Display as inverted for better UX: 3/3 = no errors
    display_score = 4 - result.correctness.score
    _render_score_card(
        title=t("paper_review.result.correctness_score"),
        score=display_score,
        max_score=3,
        icon="🔍",
        description=result.correctness.reasoning[:300] if result.correctness.reasoning else None,
    )

    # Show key issues
    if result.correctness.key_issues:
        with st.expander(t("paper_review.result.key_issues"), expanded=False):
            for i, issue in enumerate(result.correctness.key_issues, 1):
                st.markdown(f"**{i}.** {issue}")


def _render_criticality_result(result: PaperReviewResult) -> None:
    """Render criticality verification results."""
    if not result.criticality:
        return

    # Score mapping: 1 = false positives (best), 3 = major errors (worst)
    display_score = 4 - result.criticality.score
    _render_score_card(
        title=t("paper_review.result.criticality_score"),
        score=display_score,
        max_score=3,
        icon="⚠️",
        description=result.criticality.reasoning[:300] if result.criticality.reasoning else None,
    )

    issues = result.criticality.issues

    # Show categorized issues
    col1, col2 = st.columns(2)

    with col1:
        if issues.major:
            st.markdown(f"**🔴 {t('paper_review.result.major_issues')}** ({len(issues.major)})")
            for issue in issues.major[:5]:
                st.markdown(f"- {issue}")
            if len(issues.major) > 5:
                st.caption(f"... and {len(issues.major) - 5} more")

    with col2:
        if issues.minor:
            st.markdown(f"**🟡 {t('paper_review.result.minor_issues')}** ({len(issues.minor)})")
            for issue in issues.minor[:5]:
                st.markdown(f"- {issue}")
            if len(issues.minor) > 5:
                st.caption(f"... and {len(issues.minor) - 5} more")

    if issues.false_positives:
        with st.expander(f"⚪ {t('paper_review.result.false_positives')} ({len(issues.false_positives)})"):
            for issue in issues.false_positives:
                st.markdown(f"- {issue}")


def _render_bib_verification_result(result: PaperReviewResult) -> None:
    """Render bibliography verification results."""
    if not result.bib_verification:
        return

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">📚</span>
                <span style="font-weight: 600; color: #F1F5F9;">{t("paper_review.result.bib_summary")}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    for _, summary in result.bib_verification.items():
        # Verification rate
        rate = summary.verification_rate
        # Color kept for potential future use
        _ = "#22C55E" if rate >= 0.8 else ("#F59E0B" if rate >= 0.5 else "#EF4444")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("paper_review.result.total_refs"), summary.total_references)
        with col2:
            st.metric(t("paper_review.result.verified"), summary.verified)
        with col3:
            st.metric(t("paper_review.result.suspect"), summary.suspect)
        with col4:
            st.metric(t("paper_review.result.verification_rate"), f"{rate:.0%}")

        # Show suspect references
        if summary.suspect_references:
            with st.expander(f"⚠️ {t('paper_review.result.suspect_refs')} ({len(summary.suspect_references)})"):
                for ref in summary.suspect_references[:10]:
                    st.markdown(f"- {ref}")
                if len(summary.suspect_references) > 10:
                    st.caption(f"... and {len(summary.suspect_references) - 10} more")


def _render_error_state(error: str) -> None:
    """Render error state."""
    escaped_error = html.escape(error)

    st.markdown(
        f"""<div style="
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">❌</span>
                <span style="font-weight: 600; color: #EF4444;">{t("paper_review.progress.failed")}</span>
            </div>
            <div style="color: #FCA5A5; font-size: 0.9rem; line-height: 1.6;">
                {escaped_error}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_result_panel(task_result: ReviewTaskResult) -> None:
    """Render the complete result panel.

    Args:
        task_result: The review task result containing all data
    """
    if not task_result.success:
        _render_error_state(task_result.error or "Unknown error")
        return

    if not task_result.result:
        return

    result = task_result.result

    # Success header
    st.markdown(
        f"""<div style="
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <div style="font-weight: 600; color: #22C55E;">{t("paper_review.single.completed")}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem;">
                        {task_result.paper_name} • {task_result.elapsed_time_display}
                    </div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Safety check (always show first)
    _render_safety_result(result)

    # If not safe, don't show other results
    if not result.is_safe:
        st.warning("Review stopped due to safety issues. Other analyses were not performed.")
        return

    # Review score
    _render_review_result(result)

    # Correctness analysis
    _render_correctness_result(result)

    # Criticality verification
    _render_criticality_result(result)

    # Bibliography verification
    _render_bib_verification_result(result)

    # Download report
    st.markdown("---")
    if task_result.report:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"📥 {t('paper_review.result.download_report')}",
                data=task_result.report,
                file_name=f"{task_result.paper_name.replace(' ', '_')}_review.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            if st.button(
                f"📋 {t('paper_review.result.copy_report')}",
                use_container_width=True,
            ):
                st.code(task_result.report, language="markdown")
