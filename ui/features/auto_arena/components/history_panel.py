# -*- coding: utf-8 -*-
"""History panel component for Auto Arena feature.

Displays past evaluation tasks with options to view, resume, or delete.
"""

from datetime import datetime
from typing import Callable

import streamlit as st
from features.auto_arena.services.history_manager import HistoryManager, TaskSummary


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


def _render_task_card(
    task: TaskSummary,
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
    # Status styling
    status_styles = {
        "completed": {"color": "#10B981", "bg": "rgba(16, 185, 129, 0.1)", "icon": "✓", "text": "Completed"},
        "failed": {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.1)", "icon": "✕", "text": "Failed"},
        "in_progress": {"color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.1)", "icon": "●", "text": "In Progress"},
    }
    status = status_styles.get(task.status, status_styles["in_progress"])

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
                            background: {status['bg']};
                            color: {status['color']};
                            border-radius: 4px;
                            font-size: 0.7rem;
                            font-weight: 600;
                        ">{status['icon']} {status['text']}</span>
                        <span style="font-size: 0.75rem; color: #64748B;">
                            {_format_time_ago(task.created_at)}
                        </span>
                    </div>
                    <div style="
                        font-weight: 600;
                        color: #F1F5F9;
                        font-size: 0.9rem;
                        margin-bottom: 0.25rem;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    ">{task.task_description[:60]}{'...' if len(task.task_description) > 60 else ''}</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">
                        {len(task.target_models)} models • {task.num_queries} queries
                        {f' • Best: {task.best_model}' if task.best_model else ''}
                    </div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👁️ View", key=f"view_{task.task_id}", use_container_width=True):
            if on_view:
                on_view(task.task_id)
    with col2:
        can_resume = task.status == "in_progress"
        if st.button(
            "▶️ Resume",
            key=f"resume_{task.task_id}",
            use_container_width=True,
            disabled=not can_resume,
        ):
            if on_resume:
                on_resume(task.task_id)
    with col3:
        if st.button("🗑️ Delete", key=f"delete_{task.task_id}", use_container_width=True):
            if on_delete:
                on_delete(task.task_id)


def render_history_panel(
    on_view: Callable[[str], None] | None = None,
    on_resume: Callable[[str], None] | None = None,
    on_delete: Callable[[str], None] | None = None,
    limit: int = 10,
) -> None:
    """Render the history panel showing past evaluations.

    Args:
        on_view: Callback when view button clicked (receives task_id)
        on_resume: Callback when resume button clicked (receives task_id)
        on_delete: Callback when delete button clicked (receives task_id)
        limit: Maximum number of tasks to show
    """
    st.markdown(
        """<div class="section-header">
            <span style="margin-right: 0.5rem;">📜</span>Evaluation History
        </div>""",
        unsafe_allow_html=True,
    )

    # Load history
    history_manager = HistoryManager()
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
                    No evaluation history yet<br/>
                    <span style="font-size: 0.8rem;">Start an evaluation to see it here</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Render task cards
    for task in tasks:
        _render_task_card(task, on_view, on_resume, on_delete)

    # Show count
    st.markdown(
        f"""<div style="text-align: center; font-size: 0.75rem; color: #64748B; margin-top: 0.5rem;">
            Showing {len(tasks)} evaluation{'s' if len(tasks) != 1 else ''}
        </div>""",
        unsafe_allow_html=True,
    )
