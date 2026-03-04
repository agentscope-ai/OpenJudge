# -*- coding: utf-8 -*-
"""Result panel component for Auto Rubric feature.

Displays:
- Empty state when no grader is generated
- Generated grader information and rubrics
- Export, test, and copy functionality
"""

import html
from typing import Any

import streamlit as st
from features.auto_rubric.components.rubric_tester import render_test_section_compact
from features.auto_rubric.services.export_service import ExportService
from shared.i18n import t

from openjudge.graders.llm_grader import LLMGrader


def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS.

    Args:
        text: Text to escape.

    Returns:
        HTML-escaped text.
    """
    return html.escape(str(text)) if text else ""


def render_empty_state() -> None:
    """Render the empty state when no grader has been generated."""
    st.markdown(
        f"""
        <div style="
            background: rgba(30, 41, 59, 0.5);
            border: 1px dashed #475569;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔧</div>
            <div style="
                font-size: 1.1rem;
                font-weight: 600;
                color: #F1F5F9;
                margin-bottom: 0.5rem;
            ">{t('rubric.result.empty_title')}</div>
            <div style="color: #94A3B8; font-size: 0.9rem;">
                {t('rubric.result.empty_desc')}
            </div>
            <div style="
                color: #6366F1;
                font-size: 0.85rem;
                margin-top: 1rem;
            ">
                💡 {t('rubric.result.empty_tip')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grader_info(config: dict[str, Any]) -> None:
    """Render grader information card.

    Args:
        config: Grader configuration dictionary.
    """
    # Escape user-provided values to prevent XSS
    grader_name = _escape_html(config.get("grader_name", ""))
    grader_mode = _escape_html(config.get("grader_mode", "pointwise"))
    language = _escape_html(config.get("language", "EN"))
    min_score = config.get("min_score", 0)
    max_score = config.get("max_score", 5)

    # Build the info items list
    info_items = [
        f'<div><span style="color: #94A3B8;">{t("rubric.result.name_label")}:</span> '
        f'<span style="color: #F1F5F9;">{grader_name}</span></div>',
        f'<div><span style="color: #94A3B8;">{t("rubric.result.mode_label")}:</span> '
        f'<span style="color: #F1F5F9;">{grader_mode.capitalize()}</span></div>',
        f'<div><span style="color: #94A3B8;">{t("rubric.result.language_label")}:</span> '
        f'<span style="color: #F1F5F9;">{language}</span></div>',
    ]

    # Add score range only for pointwise mode
    if grader_mode.lower() == "pointwise":
        info_items.append(
            f'<div><span style="color: #94A3B8;">{t("rubric.result.score_range_label")}:</span> '
            f'<span style="color: #F1F5F9;">{min_score} - {max_score}</span></div>'
        )

    info_html = "\n".join(info_items)

    st.markdown(
        f"""
        <div style="
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-weight: 600; color: #A5B4FC; margin-bottom: 0.5rem;">
                {t('rubric.result.grader_info')}
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.9rem;">
                {info_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rubrics_display(rubrics: str) -> None:
    """Render the generated rubrics.

    Args:
        rubrics: The generated rubrics text.
    """
    st.markdown(
        f"""
        <div style="
            font-weight: 600;
            color: #F1F5F9;
            margin-bottom: 0.5rem;
        ">{t('rubric.result.rubrics_title')}</div>
        """,
        unsafe_allow_html=True,
    )

    # Use st.text_area in disabled mode for safe display of LLM-generated content
    # This avoids HTML injection risks from untrusted content
    st.text_area(
        label="rubrics_display",
        value=rubrics,
        height=300,
        disabled=True,
        label_visibility="collapsed",
        key="rubric_display_area",
    )


def render_export_section(config: dict[str, Any]) -> None:
    """Render export options section.

    Args:
        config: Grader configuration dictionary.
    """
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

    export_service = ExportService()
    grader_name = config.get("grader_name", "grader")

    # Export format selection - use stable values to survive UI language switch
    format_values = ["python", "yaml", "json"]
    format_labels = {
        "python": t("rubric.export.python"),
        "yaml": t("rubric.export.yaml"),
        "json": t("rubric.export.json"),
    }

    if "rubric_export_format" not in st.session_state:
        st.session_state["rubric_export_format"] = "python"

    format_type = st.selectbox(
        t("rubric.export.format"),
        options=format_values,
        format_func=lambda x: format_labels.get(x, x),
        key="rubric_export_format",
    )

    # Determine format type and content
    if format_type == "python":
        content = export_service.export_python(config)
        language = "python"
    elif format_type == "yaml":
        content = export_service.export_yaml(config)
        language = "yaml"
    else:
        content = export_service.export_json(config)
        language = "json"

    # Preview in expander
    with st.expander(t("rubric.export.preview"), expanded=False):
        st.code(content, language=language)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        filename = export_service.get_filename(grader_name, format_type)
        st.download_button(
            f"📥 {t('rubric.export.download')}",
            data=content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        if st.button(f"📋 {t('rubric.export.copy')}", use_container_width=True, key="rubric_copy_export"):
            st.session_state["rubric_clipboard"] = content
            st.toast(t("rubric.export.copy_success"), icon="✅")


def render_result_panel(
    result: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    grader: LLMGrader | None = None,
) -> None:
    """Render the result panel.

    Args:
        result: Generation result from RubricGeneratorService.
        config: Grader configuration dictionary.
        grader: The generated LLMGrader instance for testing.
    """
    st.markdown(
        f"""
        <div style="
            font-weight: 600;
            color: #F1F5F9;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        ">{t('rubric.result.title')}</div>
        """,
        unsafe_allow_html=True,
    )

    # If no result at all, show empty state
    if result is None:
        render_empty_state()
        return

    # Check if generation failed - show error even without config
    if not result.get("success", False):
        error = _escape_html(result.get("error", "Unknown error"))
        st.error(f"{t('rubric.result.failed')}: {error}")
        return

    # Success state but no config - shouldn't happen, but handle gracefully
    if config is None:
        render_empty_state()
        return

    # Success state
    st.success(t("rubric.result.success"))

    # Grader information
    grader_config = result.get("grader_config", config)
    render_grader_info(grader_config)

    # Display rubrics
    rubrics = result.get("rubrics", "")
    if rubrics:
        render_rubrics_display(rubrics)

        # Copy rubrics button
        if st.button(f"📋 {t('rubric.result.copy')}", key="rubric_copy_rubrics"):
            st.session_state["rubric_clipboard"] = rubrics
            st.toast(t("rubric.result.copy_success"), icon="✅")

    # Export section
    render_export_section(grader_config)

    # Test section (compact version in expander)
    if grader is not None:
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        grader_mode = grader_config.get("grader_mode", "pointwise")
        render_test_section_compact(grader=grader, grader_mode=grader_mode)
