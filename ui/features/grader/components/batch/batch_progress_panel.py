# -*- coding: utf-8 -*-
"""Progress panel for batch evaluation.

Displays real-time progress of batch evaluation including:
- Overall progress bar
- Success/failure counts
- Estimated remaining time
- Recent logs
"""


import streamlit as st
from features.grader.services.batch_runner import BatchProgress, BatchStatus
from shared.components.common import render_section_header


def _format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _get_status_display(status: BatchStatus) -> tuple[str, str, str]:
    """Get status display info.

    Returns:
        Tuple of (status_text, status_color, status_icon)
    """
    status_map = {
        BatchStatus.PENDING: ("Ready", "#64748B", "○"),
        BatchStatus.RUNNING: ("Running...", "#6366F1", "●"),
        BatchStatus.PAUSED: ("Paused", "#F59E0B", "⏸"),
        BatchStatus.COMPLETED: ("Completed", "#10B981", "✓"),
        BatchStatus.FAILED: ("Failed", "#EF4444", "✗"),
    }
    return status_map.get(status, ("Unknown", "#64748B", "?"))


def render_batch_progress_panel(
    progress: BatchProgress | None = None,
    is_running: bool = False,
) -> None:
    """Render the batch progress panel.

    Args:
        progress: Current progress state
        is_running: Whether evaluation is currently running
    """
    render_section_header("Evaluation Progress / 评估进度")

    # Empty state - not started
    if progress is None or progress.status == BatchStatus.PENDING:
        st.markdown(
            """<div style="
                text-align: center;
                padding: 2rem;
                color: #64748B;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-size: 0.9rem;">
                    Upload data and click <strong style="color: #6366F1;">Start Evaluation</strong> to begin<br/>
                    上传数据并点击<strong style="color: #6366F1;">开始评估</strong>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Get status display
    status_text, status_color, status_icon = _get_status_display(progress.status)

    # Calculate progress percentage
    if progress.total_count > 0:
        pct = (progress.completed_count / progress.total_count) * 100
    else:
        pct = 0

    # Overall progress header
    st.markdown(
        f"""<div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        ">
            <span style="color: #94A3B8; font-size: 0.85rem;">
                Overall Progress / 整体进度
            </span>
            <span style="
                color: {status_color};
                font-weight: 600;
                font-size: 0.85rem;
            ">
                {status_icon} {status_text} {pct:.1f}%
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Progress bar
    st.markdown(
        f"""<div style="
            background: rgba(100, 116, 139, 0.2);
            border-radius: 6px;
            height: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            <div style="
                background: linear-gradient(90deg, #6366F1, #8B5CF6);
                width: {pct}%;
                height: 100%;
                transition: width 0.3s ease;
            "></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div style="
                background: rgba(30, 41, 59, 0.5);
                border-radius: 8px;
                padding: 0.75rem;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 700; color: #F1F5F9;">
                    {progress.completed_count}
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">
                    / {progress.total_count}
                </div>
                <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">
                    Completed / 已完成
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div style="
                background: rgba(16, 185, 129, 0.1);
                border-radius: 8px;
                padding: 0.75rem;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10B981;">
                    {progress.success_count}
                </div>
                <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">
                    Success / 成功
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div style="
                background: rgba(239, 68, 68, 0.1);
                border-radius: 8px;
                padding: 0.75rem;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 700; color: #EF4444;">
                    {progress.failed_count}
                </div>
                <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">
                    Failed / 失败
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        elapsed_str = _format_time(progress.elapsed_seconds)
        remaining_str = _format_time(progress.estimated_remaining_seconds) if is_running else "--"
        st.markdown(
            f"""<div style="
                background: rgba(30, 41, 59, 0.5);
                border-radius: 8px;
                padding: 0.75rem;
                text-align: center;
            ">
                <div style="font-size: 1rem; font-weight: 600; color: #F1F5F9;">
                    {elapsed_str}
                </div>
                <div style="font-size: 0.7rem; color: #64748B;">
                    ~{remaining_str} left
                </div>
                <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">
                    Time / 耗时
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Error display
    if progress.error:
        st.markdown(
            f"""<div style="
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            ">
                <div style="font-weight: 600; color: #EF4444; margin-bottom: 0.5rem;">
                    ❌ Error / 错误
                </div>
                <div style="color: #FCA5A5; font-size: 0.85rem;">
                    {progress.error}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Logs section
    if progress.logs:
        with st.expander("📋 Logs / 日志", expanded=False):
            log_text = "\n".join(progress.logs[-30:])  # Show last 30 logs
            st.code(log_text, language="text")


def render_empty_progress_state() -> None:
    """Render empty progress state before evaluation starts."""
    render_section_header("Evaluation Progress / 评估进度")

    st.markdown(
        """<div style="
            text-align: center;
            padding: 2rem;
            color: #64748B;
            background: rgba(30, 41, 59, 0.3);
            border-radius: 8px;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
            <div style="font-size: 0.9rem;">
                Upload data and click <strong style="color: #6366F1;">Start Evaluation</strong> to begin<br/>
                上传数据并点击<strong style="color: #6366F1;">开始评估</strong>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
