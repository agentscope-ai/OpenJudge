# -*- coding: utf-8 -*-
"""Progress panel component for Paper Review feature.

Displays real-time progress during paper review execution.
"""

from typing import Optional

import streamlit as st
from shared.i18n import t

from cookbooks.paper_review import ReviewProgress, ReviewStage

# Stage icons mapping
STAGE_ICONS = {
    ReviewStage.NOT_STARTED: "⏸️",
    ReviewStage.LOADING_PDF: "📄",
    ReviewStage.SAFETY_CHECK: "🛡️",
    ReviewStage.CORRECTNESS: "🔍",
    ReviewStage.REVIEW: "📝",
    ReviewStage.CRITICALITY: "⚠️",
    ReviewStage.BIB_VERIFICATION: "📚",
    ReviewStage.COMPLETED: "✅",
    ReviewStage.FAILED: "❌",
}

# Stage i18n key mapping
STAGE_I18N_KEYS = {
    ReviewStage.LOADING_PDF: "paper_review.progress.loading_pdf",
    ReviewStage.SAFETY_CHECK: "paper_review.progress.safety_check",
    ReviewStage.CORRECTNESS: "paper_review.progress.correctness",
    ReviewStage.REVIEW: "paper_review.progress.review",
    ReviewStage.CRITICALITY: "paper_review.progress.criticality",
    ReviewStage.BIB_VERIFICATION: "paper_review.progress.bib_verification",
    ReviewStage.COMPLETED: "paper_review.progress.completed",
    ReviewStage.FAILED: "paper_review.progress.failed",
}


def _get_stage_display_name(stage: ReviewStage) -> str:
    """Get localized display name for a stage."""
    key = STAGE_I18N_KEYS.get(stage)
    if key:
        return t(key)
    return stage.value.replace("_", " ").title()


def _render_stage_item(
    stage: ReviewStage,
    is_current: bool = False,
    is_completed: bool = False,
    is_failed: bool = False,
) -> None:
    """Render a single stage item in the progress list."""
    icon = STAGE_ICONS.get(stage, "⏳")
    name = _get_stage_display_name(stage)

    if is_failed:
        color = "#EF4444"
        bg_color = "rgba(239, 68, 68, 0.1)"
        border_color = "rgba(239, 68, 68, 0.3)"
    elif is_completed:
        color = "#22C55E"
        bg_color = "rgba(34, 197, 94, 0.1)"
        border_color = "rgba(34, 197, 94, 0.3)"
    elif is_current:
        color = "#6366F1"
        bg_color = "rgba(99, 102, 241, 0.1)"
        border_color = "rgba(99, 102, 241, 0.3)"
    else:
        color = "#64748B"
        bg_color = "transparent"
        border_color = "#334155"

    st.markdown(
        f"""<div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: {bg_color};
            border: 1px solid {border_color};
            border-radius: 8px;
            margin-bottom: 0.5rem;
        ">
            <span style="font-size: 1.25rem;">{icon}</span>
            <span style="color: {color}; font-weight: {'600' if is_current else '400'};">
                {name}
            </span>
            {f'<span style="margin-left: auto; color: {color};">'
             f'{"⏳" if is_current else "✓" if is_completed else ""}</span>'
             if is_current or is_completed else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def render_progress_panel(progress: Optional[ReviewProgress] = None) -> None:
    """Render the progress panel showing review status.

    Args:
        progress: Current progress state, or None for empty state
    """
    if progress is None:
        _render_empty_progress_state()
        return

    # Header with progress percentage
    st.markdown(
        f"""<div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        ">
            <div style="font-weight: 600; color: #F1F5F9; font-size: 1rem;">
                {t("paper_review.single.in_progress")}
            </div>
            <div style="color: #6366F1; font-weight: 600;">
                {progress.progress_percent:.0f}%
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Progress bar
    progress_color = (
        "#22C55E"
        if progress.stage == ReviewStage.COMPLETED
        else ("#EF4444" if progress.stage == ReviewStage.FAILED else "#6366F1")
    )
    st.markdown(
        f"""<div style="
            height: 8px;
            background: #1E293B;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        ">
            <div style="
                width: {progress.progress_percent}%;
                height: 100%;
                background: {progress_color};
                border-radius: 4px;
                transition: width 0.3s ease;
            "></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Current stage info
    if progress.stage_name:
        icon = STAGE_ICONS.get(progress.stage, "⏳")
        # Translate stage_description if it's an i18n key
        stage_desc = t(progress.stage_description) if progress.stage_description else ""
        st.markdown(
            f"""<div style="
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <div>
                        <div style="font-weight: 600; color: #F1F5F9;">
                            {_get_stage_display_name(progress.stage)}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">
                            {stage_desc}
                        </div>
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Stage count
    st.markdown(
        f"""<div style="
            color: #94A3B8;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 1rem;
        ">
            {t("paper_review.batch.completed_papers")}: {progress.completed_stages} / {progress.total_stages}
        </div>""",
        unsafe_allow_html=True,
    )

    # Error message if failed
    if progress.error:
        st.markdown(
            f"""<div style="
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            ">
                <div style="color: #EF4444; font-weight: 600; margin-bottom: 0.5rem;">
                    {t("paper_review.progress.failed")}
                </div>
                <div style="color: #FCA5A5; font-size: 0.9rem;">
                    {progress.error}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_empty_progress_state() -> None:
    """Render empty state when no progress is available."""
    st.markdown(
        f"""<div style="
            text-align: center;
            padding: 3rem;
            color: #64748B;
            border: 2px dashed #334155;
            border-radius: 8px;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
            <div>{t('paper_review.single.empty_state')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_compact_progress(progress: ReviewProgress) -> str:
    """Render a compact progress string for status updates.

    Args:
        progress: Current progress state

    Returns:
        Formatted progress string
    """
    icon = STAGE_ICONS.get(progress.stage, "⏳")
    name = _get_stage_display_name(progress.stage)
    return f"{icon} {name} ({progress.progress_percent:.0f}%)"
