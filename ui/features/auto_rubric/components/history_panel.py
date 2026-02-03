# -*- coding: utf-8 -*-
"""History panel for Auto Rubric feature.

Displays list of previously generated graders and provides
view, export, and delete functionality.
"""

import html
from datetime import datetime
from typing import Any, Callable

import streamlit as st
from features.auto_rubric.services.export_service import ExportService
from features.auto_rubric.services.history_manager import HistoryManager
from shared.i18n import t


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text else ""


def _format_datetime(iso_str: str) -> str:
    """Format ISO datetime string for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str


def render_history_panel(
    on_view: Callable[[str], None] | None = None,
    on_delete: Callable[[str], None] | None = None,
    limit: int = 20,
) -> None:
    """Render the history panel showing previously generated graders.

    Args:
        on_view: Callback when "View" is clicked, receives task_id.
        on_delete: Callback when "Delete" is clicked, receives task_id.
        limit: Maximum number of tasks to display.
    """
    st.markdown(
        f"""
        <div style="
            font-weight: 600;
            color: #F1F5F9;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        ">{t('rubric.history.title')}</div>
        """,
        unsafe_allow_html=True,
    )

    # Load history
    history_manager = HistoryManager()
    tasks = history_manager.list_tasks(limit=limit)

    if not tasks:
        # Empty state
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                border: 1px dashed #475569;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📜</div>
                <div style="color: #94A3B8; font-size: 0.9rem;">
                    {t('rubric.history.empty')}
                </div>
                <div style="color: #64748B; font-size: 0.85rem; margin-top: 0.5rem;">
                    {t('rubric.history.empty_hint')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Show count
    st.markdown(
        f"<div style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.75rem;'>"
        f"{t('rubric.history.count', count=len(tasks))}</div>",
        unsafe_allow_html=True,
    )

    # Render task list
    for task in tasks:
        _render_task_card(task, on_view, on_delete)


def _render_task_card(
    task: dict[str, Any],
    on_view: Callable[[str], None] | None,
    on_delete: Callable[[str], None] | None,
) -> None:
    """Render a single task card."""
    task_id = task.get("task_id", "")
    grader_name = _escape_html(task.get("grader_name", "Unnamed"))
    mode = task.get("mode", "simple")
    grader_mode = task.get("grader_mode", "pointwise")
    data_count = task.get("data_count")
    created_at = _format_datetime(task.get("created_at", ""))

    # Mode badge
    mode_badge = "⚡ Simple" if mode == "simple" else "📊 Iterative"
    mode_color = "#A5B4FC" if mode == "simple" else "#34D399"

    # Data count display
    data_info = ""
    if data_count and mode == "iterative":
        data_info = f" | {data_count} records"

    with st.container():
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="
                            font-weight: 600;
                            color: #F1F5F9;
                            font-size: 1rem;
                            margin-bottom: 0.25rem;
                        ">🔧 {grader_name}</div>
                        <div style="
                            font-size: 0.85rem;
                            color: #94A3B8;
                        ">
                            <span style="color: {mode_color};">{mode_badge}</span>
                            | {grader_mode.capitalize()}{data_info}
                        </div>
                    </div>
                    <div style="
                        color: #64748B;
                        font-size: 0.8rem;
                    ">{created_at}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button(
                f"👁️ {t('rubric.history.view')}",
                key=f"view_{task_id}",
                use_container_width=True,
            ):
                if on_view:
                    on_view(task_id)

        with col2:
            if st.button(
                f"📤 {t('rubric.history.export')}",
                key=f"export_{task_id}",
                use_container_width=True,
            ):
                st.session_state[f"rubric_export_task_{task_id}"] = True
                st.rerun()

        with col3:
            if st.button(
                f"🗑️ {t('rubric.history.delete')}",
                key=f"delete_{task_id}",
                use_container_width=True,
            ):
                if on_delete:
                    on_delete(task_id)

        # Handle export modal
        if st.session_state.get(f"rubric_export_task_{task_id}"):
            _render_export_modal(task_id)


def _render_export_modal(task_id: str) -> None:
    """Render export options for a task."""
    history_manager = HistoryManager()
    details = history_manager.get_task_details(task_id)

    if not details:
        st.error(t("rubric.history.task_not_found"))
        return

    grader_config = details.get("grader_config", {})

    with st.expander(f"📦 {t('rubric.export.title')}", expanded=True):
        export_service = ExportService()
        grader_name = grader_config.get("grader_name", "grader")

        # Use stable values to survive UI language switch
        format_values = ["python", "yaml", "json"]
        format_labels = {
            "python": t("rubric.export.python"),
            "yaml": t("rubric.export.yaml"),
            "json": t("rubric.export.json"),
        }

        export_key = f"export_format_{task_id}"
        if export_key not in st.session_state:
            st.session_state[export_key] = "python"

        format_type = st.selectbox(
            t("rubric.export.format"),
            options=format_values,
            format_func=lambda x: format_labels.get(x, x),
            key=export_key,
        )

        if format_type == "python":
            content = export_service.export_python(grader_config)
        elif format_type == "yaml":
            content = export_service.export_yaml(grader_config)
        else:
            content = export_service.export_json(grader_config)

        filename = export_service.get_filename(grader_name, format_type)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                f"📥 {t('rubric.export.download')}",
                data=content,
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            if st.button(
                f"✕ {t('common.close')}",
                key=f"close_export_{task_id}",
                use_container_width=True,
            ):
                del st.session_state[f"rubric_export_task_{task_id}"]
                st.rerun()


def render_task_detail(
    task_id: str,
    on_back: Callable[[], None] | None = None,
) -> None:
    """Render detailed view of a single task.

    Args:
        task_id: Task identifier.
        on_back: Callback for back button.
    """
    history_manager = HistoryManager()
    details = history_manager.get_task_details(task_id)

    if not details:
        st.error(t("rubric.history.task_not_found"))
        if on_back:
            if st.button(f"← {t('rubric.history.back')}"):
                on_back()
        return

    config = details.get("config", {})
    rubrics = details.get("rubrics", "")
    grader_config = details.get("grader_config", {})

    # Header with back button
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button(f"← {t('rubric.history.back')}"):
            if on_back:
                on_back()
    with col_title:
        st.markdown(
            f"""
            <div style="
                font-weight: 600;
                color: #F1F5F9;
                font-size: 1.2rem;
            ">🔧 {_escape_html(config.get('grader_name', 'Grader'))}</div>
            """,
            unsafe_allow_html=True,
        )

    # Info card
    st.markdown(
        f"""
        <div style="
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.9rem;">
                <div><span style="color: #94A3B8;">{t('rubric.result.mode_label')}:</span>
                    <span style="color: #F1F5F9;">{_escape_html(config.get('mode', 'simple')).capitalize()}</span></div>
                <div><span style="color: #94A3B8;">{t('rubric.result.language_label')}:</span>
                    <span style="color: #F1F5F9;">{_escape_html(config.get('language', 'EN'))}</span></div>
                <div><span style="color: #94A3B8;">{t('rubric.result.score_range_label')}:</span>
                    <span style="color: #F1F5F9;">{config.get('min_score', 0)}-{config.get('max_score', 5)}</span></div>
                <div><span style="color: #94A3B8;">Created:</span>
                    <span style="color: #F1F5F9;">{_format_datetime(config.get('created_at', ''))}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rubrics display
    if rubrics:
        st.markdown(
            f"""
            <div style="
                font-weight: 600;
                color: #F1F5F9;
                margin: 1rem 0 0.5rem;
            ">{t('rubric.result.rubrics_title')}</div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            label="rubrics",
            value=rubrics,
            height=300,
            disabled=True,
            label_visibility="collapsed",
            key=f"rubric_detail_{task_id}",
        )

    # Export section
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

    export_service = ExportService()
    grader_name = config.get("grader_name", "grader")

    # Use stable values to survive UI language switch
    format_values = ["python", "yaml", "json"]
    format_labels = {
        "python": t("rubric.export.python"),
        "yaml": t("rubric.export.yaml"),
        "json": t("rubric.export.json"),
    }

    detail_export_key = f"detail_export_format_{task_id}"
    if detail_export_key not in st.session_state:
        st.session_state[detail_export_key] = "python"

    lang = st.selectbox(
        t("rubric.export.format"),
        options=format_values,
        format_func=lambda x: format_labels.get(x, x),
        key=detail_export_key,
    )

    if lang == "python":
        content = export_service.export_python(grader_config)
    elif lang == "yaml":
        content = export_service.export_yaml(grader_config)
    else:
        content = export_service.export_json(grader_config)

    with st.expander(t("rubric.export.preview"), expanded=False):
        st.code(content, language=lang)

    filename = export_service.get_filename(grader_name, lang)
    st.download_button(
        f"📥 {t('rubric.export.download')}",
        data=content,
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
        key=f"history_download_{task_id}_{lang}",
    )
