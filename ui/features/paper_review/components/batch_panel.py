# -*- coding: utf-8 -*-
"""Batch review panel component for Paper Review feature.

Provides UI for batch paper review with progress tracking.
"""

from typing import Any, Optional

import streamlit as st
from features.paper_review.services.batch_runner import (
    BatchProgress,
    BatchResult,
    BatchStatus,
)
from shared.i18n import t


def render_batch_progress(progress: Optional[BatchProgress]) -> None:
    """Render batch progress panel.

    Args:
        progress: Current batch progress state
    """
    if progress is None:
        _render_empty_batch_state()
        return

    # Status indicator
    status_config = _get_status_config(progress.status)

    st.markdown(
        f"""<div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        ">
            <div style="font-weight: 600; color: #F1F5F9; font-size: 1rem;">
                {t("paper_review.batch.progress_title")}
            </div>
            <div style="
                padding: 0.25rem 0.75rem;
                background: {status_config['bg']};
                border: 1px solid {status_config['border']};
                border-radius: 12px;
                color: {status_config['color']};
                font-size: 0.8rem;
                font-weight: 500;
            ">
                {status_config['icon']} {status_config['text']}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Progress bar
    pct = progress.progress_percent
    bar_color = status_config["color"]

    st.markdown(
        f"""<div style="
            height: 10px;
            background: #1E293B;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            <div style="
                width: {pct}%;
                height: 100%;
                background: {bar_color};
                border-radius: 5px;
                transition: width 0.3s ease;
            "></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(t("paper_review.batch.total_papers"), progress.total)
    with col2:
        st.metric(t("paper_review.batch.completed_papers"), progress.completed)
    with col3:
        st.metric(
            "Failed",
            progress.failed,
            delta=None if progress.failed == 0 else f"-{progress.failed}",
            delta_color="inverse",
        )

    # Current paper
    if progress.current_paper:
        st.markdown(
            f"""<div style="
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-top: 1rem;
            ">
                <div style="color: #94A3B8; font-size: 0.85rem;">
                    {t("paper_review.batch.current_paper")}
                </div>
                <div style="color: #F1F5F9; font-weight: 500; margin-top: 0.25rem;">
                    📄 {progress.current_paper}
                </div>
                <div style="
                    height: 4px;
                    background: #1E293B;
                    border-radius: 2px;
                    overflow: hidden;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {progress.current_paper_progress}%;
                        height: 100%;
                        background: #6366F1;
                        border-radius: 2px;
                    "></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _get_status_config(status: BatchStatus) -> dict[str, str]:
    """Get display configuration for a batch status."""
    configs = {
        BatchStatus.PENDING: {
            "icon": "⏸️",
            "text": "Pending",
            "color": "#64748B",
            "bg": "rgba(100, 116, 139, 0.1)",
            "border": "rgba(100, 116, 139, 0.3)",
        },
        BatchStatus.RUNNING: {
            "icon": "🔄",
            "text": "Running",
            "color": "#6366F1",
            "bg": "rgba(99, 102, 241, 0.1)",
            "border": "rgba(99, 102, 241, 0.3)",
        },
        BatchStatus.COMPLETED: {
            "icon": "✅",
            "text": "Completed",
            "color": "#22C55E",
            "bg": "rgba(34, 197, 94, 0.1)",
            "border": "rgba(34, 197, 94, 0.3)",
        },
        BatchStatus.FAILED: {
            "icon": "❌",
            "text": "Failed",
            "color": "#EF4444",
            "bg": "rgba(239, 68, 68, 0.1)",
            "border": "rgba(239, 68, 68, 0.3)",
        },
        BatchStatus.CANCELLED: {
            "icon": "⏹️",
            "text": "Cancelled",
            "color": "#F59E0B",
            "bg": "rgba(245, 158, 11, 0.1)",
            "border": "rgba(245, 158, 11, 0.3)",
        },
    }
    return configs.get(status, configs[BatchStatus.PENDING])


def _render_empty_batch_state() -> None:
    """Render empty state for batch review."""
    st.markdown(
        f"""<div style="
            text-align: center;
            padding: 3rem;
            color: #64748B;
            border: 2px dashed #334155;
            border-radius: 8px;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📦</div>
            <div>{t('paper_review.batch.upload_help')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_batch_results(batch_result: BatchResult) -> None:
    """Render batch review results.

    Args:
        batch_result: The batch review result
    """
    if not batch_result:
        return

    # Summary header
    success_rate = (batch_result.completed / batch_result.total * 100) if batch_result.total > 0 else 0
    header_color = "#22C55E" if success_rate >= 80 else ("#F59E0B" if success_rate >= 50 else "#EF4444")

    st.markdown(
        f"""<div style="
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-weight: 600; color: #F1F5F9; font-size: 1.1rem;">
                        Batch Review Complete
                    </div>
                    <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 0.25rem;">
                        {batch_result.completed}/{batch_result.total} papers reviewed
                        • {batch_result.elapsed_time_display}
                    </div>
                </div>
                <div style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: {header_color};
                ">
                    {success_rate:.0f}%
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Results table
    if batch_result.results:
        for i, result in enumerate(batch_result.results):
            _render_result_row(i + 1, result)


def _render_result_row(index: int, result: Any) -> None:
    """Render a single result row.

    Args:
        index: Row index
        result: ReviewTaskResult
    """
    success = result.success
    icon = "✅" if success else "❌"
    # Colors kept for potential future use
    _ = "rgba(34, 197, 94, 0.05)" if success else "rgba(239, 68, 68, 0.05)"  # bg_color
    _ = "rgba(34, 197, 94, 0.2)" if success else "rgba(239, 68, 68, 0.2)"  # border_color

    review_score = "-"
    correctness_score = "-"

    if result.result:
        if result.result.review:
            review_score = f"{result.result.review.score}/6"
        if result.result.correctness:
            display_score = 4 - result.result.correctness.score
            correctness_score = f"{display_score}/3"

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1.5, 1.5, 1.5])

        with col1:
            st.markdown(f"**{index}**")
        with col2:
            st.markdown(f"{icon} **{result.paper_name}**")
        with col3:
            st.caption(f"📝 {review_score}")
        with col4:
            st.caption(f"🔍 {correctness_score}")
        with col5:
            st.caption(result.elapsed_time_display)

        # Show error if failed
        if not success and result.error:
            st.error(result.error, icon="⚠️")

        # Download report button
        if success and result.report:
            st.download_button(
                label="📥 Report",
                data=result.report,
                file_name=f"{result.paper_name.replace(' ', '_')}_review.md",
                mime="text/markdown",
                key=f"batch_download_{index}",
            )
