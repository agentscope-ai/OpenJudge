# -*- coding: utf-8 -*-
"""History panel component for Batch Grader Evaluation.

Displays past batch evaluation tasks with options to view, resume, or delete.
"""

from datetime import datetime
from typing import Callable

import streamlit as st
from features.grader.services.batch_history_manager import (
    BatchHistoryManager,
    BatchTaskSummary,
)


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time string.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable time ago string
    """
    now = datetime.now()
    diff = now - dt

    if diff.days > 30:
        return dt.strftime("%Y-%m-%d")
    elif diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds > 60:
        mins = diff.seconds // 60
        return f"{mins}m ago"
    else:
        return "Just now"


def _get_status_style(status: str) -> dict[str, str]:
    """Get status styling.

    Args:
        status: Task status string

    Returns:
        Style dictionary with color, bg, icon, text
    """
    status_styles = {
        "completed": {
            "color": "#10B981",
            "bg": "rgba(16, 185, 129, 0.1)",
            "icon": "✓",
            "text": "Completed / 已完成",
        },
        "failed": {
            "color": "#EF4444",
            "bg": "rgba(239, 68, 68, 0.1)",
            "icon": "✕",
            "text": "Failed / 失败",
        },
        "running": {
            "color": "#6366F1",
            "bg": "rgba(99, 102, 241, 0.1)",
            "icon": "●",
            "text": "Running / 运行中",
        },
        "paused": {
            "color": "#F59E0B",
            "bg": "rgba(245, 158, 11, 0.1)",
            "icon": "⏸",
            "text": "Paused / 已暂停",
        },
        "pending": {
            "color": "#64748B",
            "bg": "rgba(100, 116, 139, 0.1)",
            "icon": "○",
            "text": "Pending / 等待中",
        },
    }
    return status_styles.get(status, status_styles["pending"])


def _render_task_card(
    task: BatchTaskSummary,
    on_view: Callable[[str], None] | None = None,
    on_resume: Callable[[str], None] | None = None,
    on_delete: Callable[[str], None] | None = None,
) -> None:
    """Render a single task card.

    Args:
        task: Task summary to display
        on_view: Callback when view button clicked
        on_resume: Callback when resume button clicked
        on_delete: Callback when delete button clicked
    """
    style = _get_status_style(task.status)

    # Progress percentage
    progress_pct = (task.completed_count / task.total_count * 100) if task.total_count > 0 else 0

    # Stats display
    stats_parts = [f"{task.total_count} items"]
    if task.avg_score is not None:
        stats_parts.append(f"Avg: {task.avg_score:.2f}")
    if task.pass_rate is not None:
        stats_parts.append(f"Pass: {task.pass_rate * 100:.0f}%")

    # Card container
    st.markdown(
        f"""<div style="
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(100, 116, 139, 0.2);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="
                            display: inline-flex;
                            align-items: center;
                            gap: 0.25rem;
                            padding: 0.125rem 0.5rem;
                            background: {style['bg']};
                            color: {style['color']};
                            border-radius: 4px;
                            font-size: 0.7rem;
                            font-weight: 600;
                        ">{style['icon']} {style['text']}</span>
                        <span style="font-size: 0.75rem; color: #64748B;">
                            {_format_time_ago(task.created_at)}
                        </span>
                    </div>
                    <div style="
                        font-weight: 600;
                        color: #F1F5F9;
                        font-size: 0.9rem;
                        margin-bottom: 0.25rem;
                    ">
                        {task.grader_name} ({task.grader_name_zh})
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">
                        {' • '.join(stats_parts)}
                    </div>
                </div>
                <div style="text-align: right; min-width: 60px;">
                    <div style="
                        font-size: 1.25rem;
                        font-weight: 700;
                        color: {style['color']};
                    ">{progress_pct:.0f}%</div>
                    <div style="font-size: 0.65rem; color: #64748B;">
                        {task.completed_count}/{task.total_count}
                    </div>
                </div>
            </div>

            <!-- Progress bar -->
            <div style="
                margin-top: 0.75rem;
                background: rgba(100, 116, 139, 0.2);
                border-radius: 4px;
                height: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress_pct}%;
                    height: 100%;
                    background: {style['color']};
                "></div>
            </div>

            <!-- Success/Failed counts -->
            <div style="
                display: flex;
                gap: 1rem;
                margin-top: 0.5rem;
                font-size: 0.7rem;
            ">
                <span style="color: #10B981;">✓ {task.success_count} success</span>
                <span style="color: #EF4444;">✕ {task.failed_count} failed</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👁️ View", key=f"batch_view_{task.task_id}", use_container_width=True):
            if on_view:
                on_view(task.task_id)

    with col2:
        can_resume = task.status in ("paused", "running") and task.completed_count < task.total_count
        if st.button(
            "▶️ Resume",
            key=f"batch_resume_{task.task_id}",
            use_container_width=True,
            disabled=not can_resume,
        ):
            if on_resume:
                on_resume(task.task_id)

    with col3:
        if st.button("🗑️ Delete", key=f"batch_delete_{task.task_id}", use_container_width=True):
            if on_delete:
                on_delete(task.task_id)


def render_batch_history_panel(
    on_view: Callable[[str], None] | None = None,
    on_resume: Callable[[str], None] | None = None,
    on_delete: Callable[[str], None] | None = None,
    limit: int = 15,
) -> None:
    """Render the batch history panel showing past evaluations.

    Args:
        on_view: Callback when view button clicked (receives task_id)
        on_resume: Callback when resume button clicked (receives task_id)
        on_delete: Callback when delete button clicked (receives task_id)
        limit: Maximum number of tasks to show
    """
    st.markdown(
        """<div class="section-header">
            <span style="margin-right: 0.5rem;">📜</span>Batch Evaluation History / 批量评估历史
        </div>""",
        unsafe_allow_html=True,
    )

    # Load history
    history_manager = BatchHistoryManager()
    tasks = history_manager.list_tasks(limit=limit)

    if not tasks:
        st.markdown(
            """<div style="
                text-align: center;
                padding: 2rem;
                color: #64748B;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
                <div style="font-size: 0.9rem;">
                    No batch evaluation history yet<br/>
                    暂无批量评估历史
                </div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Start a batch evaluation to see it here<br/>
                    开始批量评估后将显示在此处
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Check for incomplete tasks
    incomplete_tasks = [t for t in tasks if t.status in ("paused", "running")]
    if incomplete_tasks:
        st.markdown(
            f"""<div style="
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-bottom: 1rem;
                font-size: 0.85rem;
            ">
                <span style="color: #F59E0B;">⚠️</span>
                <span style="color: #FCD34D;">
                    {len(incomplete_tasks)} incomplete task(s) can be resumed
                    / {len(incomplete_tasks)} 个未完成的任务可以续传
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Render task cards
    for task in tasks:
        _render_task_card(task, on_view, on_resume, on_delete)

    # Show count
    st.markdown(
        f"""<div style="text-align: center; font-size: 0.75rem; color: #64748B; margin-top: 0.5rem;">
            Showing {len(tasks)} evaluation{'s' if len(tasks) != 1 else ''} / 显示 {len(tasks)} 条记录
        </div>""",
        unsafe_allow_html=True,
    )


def render_batch_task_detail(
    task_id: str,
    on_back: Callable[[], None] | None = None,
) -> None:
    """Render detailed view of a batch task.

    Args:
        task_id: Task ID to display
        on_back: Callback when back button clicked
    """
    history_manager = BatchHistoryManager()
    details = history_manager.get_task_details(task_id)

    if not details:
        st.error(f"Task not found: {task_id}")
        if on_back:
            if st.button("← Back to History"):
                on_back()
        return

    # Back button
    if on_back:
        if st.button("← Back to History / 返回历史"):
            on_back()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Task info
    config = details.get("config", {})
    summary = details.get("summary", {})

    st.markdown(
        f"""<div style="
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.5rem;">
                Task: {task_id}
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Grader: {config.get('grader_name', 'Unknown')} ({config.get('grader_name_zh', '')})
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Created: {config.get('created_at', 'Unknown')}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Import and render result panel
    from features.grader.components.batch.batch_result_panel import (
        render_batch_result_panel,
    )

    # Get score range from grader config
    grader_config = config.get("grader_config", {})
    score_range = grader_config.get("score_range", (0, 1))

    render_batch_result_panel(
        task_id=task_id,
        summary=summary,
        score_range=tuple(score_range),
        history_manager=history_manager,
    )
