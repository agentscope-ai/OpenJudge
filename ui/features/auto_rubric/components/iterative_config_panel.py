# -*- coding: utf-8 -*-
"""Iterative Rubric configuration panel for Auto Rubric feature.

Provides UI for configuring Iterative Rubric generation:
- Grader name (required)
- Data upload (required)
- Task description (optional)
- Advanced settings (optional)
"""

from typing import Any

import streamlit as st
from features.auto_rubric.components.data_upload_panel import render_data_upload_panel
from shared.i18n import t


def _render_section_header(title: str, required: bool = False, icon: str = "") -> None:
    """Render a section header with optional required badge."""
    badge = (
        '<span style="background: #EF4444; color: white; font-size: 0.7rem; '
        'padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.5rem;">'
        f'{t("rubric.config.required")}</span>'
        if required
        else '<span style="background: #64748B; color: white; font-size: 0.7rem; '
        'padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.5rem;">'
        f'{t("rubric.config.optional")}</span>'
    )

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        ">
            <span style="font-size: 1.1rem; margin-right: 0.5rem;">{icon}</span>
            <span style="font-weight: 600; color: #F1F5F9; font-size: 0.95rem;">{title}</span>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_iterative_config_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the Iterative Rubric configuration panel.

    Args:
        sidebar_config: Configuration from the sidebar.

    Returns:
        Complete configuration dictionary including:
        - grader_name: Name for the generated grader
        - dataset: Parsed training data
        - task_description: Optional task description
        - enable_categorization: Whether to group rubrics
        - categories_number: Number of categories
        - All sidebar config values
    """
    config: dict[str, Any] = {}
    config.update(sidebar_config)

    # =========================================================================
    # Section 1: Grader Name (Required)
    # =========================================================================
    _render_section_header(t("rubric.config.grader_name"), required=True, icon="📛")

    grader_name = st.text_input(
        t("rubric.config.grader_name"),
        placeholder=t("rubric.config.grader_name_placeholder"),
        help=t("rubric.config.grader_name_help"),
        key="rubric_iterative_grader_name",
        label_visibility="collapsed",
    )

    # =========================================================================
    # Section 2: Training Data (Required)
    # =========================================================================
    mode = sidebar_config.get("grader_mode", "pointwise")
    if mode == "pointwise":
        required_fields = "query, response, label_score"
    else:
        required_fields = "query, responses, label_rank"

    _render_section_header(t("rubric.upload.title"), required=True, icon="📊")

    upload_result = render_data_upload_panel(mode=mode, required_fields=required_fields)

    # =========================================================================
    # Section 3: Task Description (Optional)
    # =========================================================================
    _render_section_header(t("rubric.config.task_description"), required=False, icon="📝")

    st.markdown(
        f"<div style='color: #64748B; font-size: 0.8rem; margin-bottom: 0.5rem;'>"
        f"{t('rubric.iterative.task_desc_help')}</div>",
        unsafe_allow_html=True,
    )

    task_description = st.text_area(
        t("rubric.config.task_description"),
        placeholder=t("rubric.iterative.task_desc_placeholder"),
        height=80,
        key="rubric_iterative_task_desc",
        label_visibility="collapsed",
    )

    # =========================================================================
    # Section 4: Advanced Settings (Optional, collapsed by default)
    # =========================================================================
    with st.expander(f"⚙️ {t('rubric.iterative.advanced')} ({t('rubric.config.optional')})", expanded=False):
        st.markdown(
            f"<div style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.75rem;'>"
            f"{t('rubric.iterative.advanced_desc')}</div>",
            unsafe_allow_html=True,
        )

        enable_categorization = st.checkbox(
            t("rubric.iterative.enable_categorization"),
            value=True,
            help=t("rubric.iterative.enable_categorization_help"),
            key="rubric_enable_categorization",
        )

        col1, col2 = st.columns(2)

        with col1:
            if enable_categorization:
                categories_number = st.number_input(
                    t("rubric.iterative.categories_number"),
                    min_value=2,
                    max_value=10,
                    value=5,
                    help=t("rubric.iterative.categories_number_help"),
                    key="rubric_categories_number",
                )
            else:
                categories_number = 5

        with col2:
            query_specific_number = st.number_input(
                t("rubric.iterative.query_specific_number"),
                min_value=1,
                max_value=5,
                value=2,
                help=t("rubric.iterative.query_specific_number_help"),
                key="rubric_query_specific_number",
            )

    # =========================================================================
    # Build config
    # =========================================================================
    config["grader_name"] = grader_name
    config["dataset"] = upload_result.get("data")
    config["data_count"] = upload_result.get("count", 0)
    config["data_valid"] = upload_result.get("is_valid", False)
    config["task_description"] = task_description if task_description else None
    config["enable_categorization"] = enable_categorization
    config["categories_number"] = categories_number
    config["query_specific_generate_number"] = query_specific_number

    # Score range from data (for pointwise mode)
    config["min_score"] = upload_result.get("min_score")
    config["max_score"] = upload_result.get("max_score")

    return config


def validate_iterative_config(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate Iterative Rubric configuration.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not config.get("grader_name", "").strip():
        return False, t("rubric.validation.name_required")

    if not config.get("data_valid", False):
        return False, t("rubric.validation.data_required")

    if not config.get("api_key", "").strip():
        return False, t("rubric.validation.api_key_required")

    if not config.get("model_name", "").strip():
        return False, t("rubric.validation.model_required")

    # Check minimum data count
    data_count = config.get("data_count", 0)
    if data_count < 10:
        return False, t("rubric.validation.min_data_required", count=10)

    return True, ""
