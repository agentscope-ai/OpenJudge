# -*- coding: utf-8 -*-
"""History panel component for Paper Review feature.

Provides UI for viewing and managing review history.
"""

from typing import Callable, Optional

import streamlit as st
from features.paper_review.services.history_service import HistoryEntry
from shared.i18n import t


def render_history_list(
    entries: list[HistoryEntry],
    on_view: Optional[Callable[[str], None]] = None,  # pylint: disable=unused-argument
    on_delete: Optional[Callable[[str], None]] = None,  # pylint: disable=unused-argument
) -> Optional[str]:
    """Render the history list.

    Args:
        entries: List of history entries
        on_view: Callback when view button is clicked (reserved for future use)
        on_delete: Callback when delete button is clicked (reserved for future use)

    Returns:
        Task ID if an entry is selected for viewing
    """
    if not entries:
        _render_empty_history_state()
        return None

    selected_task_id = None

    for entry in entries:
        task_id = _render_history_entry(entry)
        if task_id:
            selected_task_id = task_id

    return selected_task_id


def _render_history_entry(entry: HistoryEntry) -> Optional[str]:
    """Render a single history entry.

    Args:
        entry: The history entry to render

    Returns:
        Task ID if view button was clicked
    """
    success = entry.success
    icon = "✅" if success else "❌"
    bg_color = "rgba(30, 41, 59, 0.6)"
    border_color = "#334155"

    # Build score display
    scores = []
    if entry.review_score is not None:
        scores.append(f"📝 {entry.review_score}/6")
    if entry.correctness_score is not None:
        scores.append(f"🔍 {entry.correctness_score}/3")
    score_text = " • ".join(scores) if scores else ""

    with st.container():
        col_main, col_actions = st.columns([4, 1])

        with col_main:
            st.markdown(
                f"""<div style="
                    background: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.1rem;">{icon}</span>
                            <span style="font-weight: 500; color: #F1F5F9;">{entry.paper_name}</span>
                        </div>
                        <div style="color: #64748B; font-size: 0.8rem;">
                            {entry.created_at_display}
                        </div>
                    </div>
                    <div style="
                        color: #94A3B8;
                        font-size: 0.85rem;
                        margin-top: 0.5rem;
                        display: flex;
                        gap: 1rem;
                    ">
                        <span>{score_text}</span>
                        <span>⏱️ {entry.elapsed_time_display}</span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_actions:
            view_clicked = st.button(
                f"👁️ {t('paper_review.history.view')}",
                key=f"view_{entry.task_id}",
                use_container_width=True,
            )

            delete_clicked = st.button(
                f"🗑️ {t('paper_review.history.delete')}",
                key=f"delete_{entry.task_id}",
                use_container_width=True,
            )

            if delete_clicked:
                return f"delete:{entry.task_id}"

            if view_clicked:
                return f"view:{entry.task_id}"

    return None


def _render_empty_history_state() -> None:
    """Render empty state for history."""
    st.markdown(
        f"""<div style="
            text-align: center;
            padding: 3rem;
            color: #64748B;
            border: 2px dashed #334155;
            border-radius: 8px;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📜</div>
            <div>{t('paper_review.history.no_history')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_history_detail(entry: HistoryEntry) -> None:
    """Render detailed view of a history entry.

    Args:
        entry: The history entry to display
    """
    # Header
    success = entry.success
    status_icon = "✅" if success else "❌"
    status_text = t("paper_review.single.completed") if success else t("paper_review.progress.failed")
    status_color = "#22C55E" if success else "#EF4444"
    status_bg = "rgba(34, 197, 94, 0.1)" if success else "rgba(239, 68, 68, 0.1)"
    status_border = "rgba(34, 197, 94, 0.3)" if success else "rgba(239, 68, 68, 0.3)"

    st.markdown(
        f"""<div style="
            background: {status_bg};
            border: 1px solid {status_border};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{status_icon}</span>
                <div>
                    <div style="font-weight: 600; color: {status_color};">{status_text}</div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">
                        {entry.paper_name} • {entry.created_at_display} • {entry.elapsed_time_display}
                    </div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Scores (if available)
    if success:
        col1, col2 = st.columns(2)

        with col1:
            if entry.review_score is not None:
                _render_score_badge(
                    t("paper_review.result.review_score"),
                    entry.review_score,
                    6,
                    "📝",
                )

        with col2:
            if entry.correctness_score is not None:
                _render_score_badge(
                    t("paper_review.result.correctness_score"),
                    entry.correctness_score,
                    3,
                    "🔍",
                )

    # Error (if failed)
    if not success and entry.error:
        st.error(entry.error)

    # Report (if available)
    if entry.report:
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"📥 {t('paper_review.result.download_report')}",
                data=entry.report,
                file_name=f"{entry.paper_name.replace(' ', '_')}_review.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"download_history_{entry.task_id}",
            )
        with col2:
            with st.expander(f"📋 {t('paper_review.result.copy_report')}"):
                st.code(entry.report, language="markdown")


def _render_score_badge(
    label: str,
    score: int,
    max_score: int,
    icon: str,
) -> None:
    """Render a score badge."""
    pct = score / max_score
    color = "#22C55E" if pct >= 0.7 else ("#F59E0B" if pct >= 0.4 else "#EF4444")

    st.markdown(
        f"""<div style="
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span>{icon}</span>
                <span style="color: #94A3B8; font-size: 0.9rem;">{label}</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {color};">
                {score} <span style="font-size: 1rem; color: #64748B;">/ {max_score}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_history_stats(stats: dict) -> None:
    """Render history statistics.

    Args:
        stats: Statistics dictionary from history service
    """
    if stats["total"] == 0:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reviews", stats["total"])
    with col2:
        st.metric("Successful", stats["success"])
    with col3:
        avg_review = f"{stats['avg_review_score']:.1f}" if stats["avg_review_score"] else "-"
        st.metric("Avg Review Score", avg_review)
    with col4:
        avg_correct = f"{stats['avg_correctness_score']:.1f}" if stats["avg_correctness_score"] else "-"
        st.metric("Avg Correctness", avg_correct)
