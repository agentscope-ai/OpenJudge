# -*- coding: utf-8 -*-
"""Simple Rubric configuration panel for Auto Rubric feature.

Provides UI for configuring Simple Rubric generation:
- Grader name
- Task description
- Usage scenario (optional)
- Sample queries (optional)
"""

from typing import Any

import streamlit as st
from shared.i18n import t


def render_simple_config_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the Simple Rubric configuration panel.

    Args:
        sidebar_config: Configuration from the sidebar.

    Returns:
        Complete configuration dictionary including:
        - grader_name: Name for the generated grader
        - task_description: Description of the task
        - scenario: Optional usage scenario
        - sample_queries: Optional list of sample queries
        - All sidebar config values
    """
    config: dict[str, Any] = {}
    config.update(sidebar_config)

    # Grader Name
    grader_name = st.text_input(
        t("rubric.config.grader_name"),
        placeholder=t("rubric.config.grader_name_placeholder"),
        help=t("rubric.config.grader_name_help"),
        key="rubric_grader_name",
    )

    # Task Description (required)
    task_description = st.text_area(
        f"{t('rubric.config.task_description')} *",
        placeholder=t("rubric.config.task_description_placeholder"),
        height=150,
        help=t("rubric.config.task_description_help"),
        key="rubric_task_description",
    )

    # Usage Scenario (optional)
    scenario = st.text_input(
        t("rubric.config.scenario"),
        placeholder=t("rubric.config.scenario_placeholder"),
        help=t("rubric.config.scenario_help"),
        key="rubric_scenario",
    )

    # Sample Queries (optional)
    sample_queries_text = st.text_area(
        t("rubric.config.sample_queries"),
        placeholder=t("rubric.config.sample_queries_placeholder"),
        height=100,
        help=t("rubric.config.sample_queries_help"),
        key="rubric_sample_queries",
    )

    # Parse sample queries (one per line)
    sample_queries = None
    if sample_queries_text.strip():
        sample_queries = [q.strip() for q in sample_queries_text.strip().split("\n") if q.strip()]

    config["grader_name"] = grader_name
    config["task_description"] = task_description
    config["scenario"] = scenario if scenario else None
    config["sample_queries"] = sample_queries

    return config


def validate_simple_config(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate Simple Rubric configuration.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.
    """
    if not config.get("grader_name", "").strip():
        return False, t("rubric.validation.name_required")

    if not config.get("task_description", "").strip():
        return False, t("rubric.validation.task_required")

    if not config.get("api_key", "").strip():
        return False, t("rubric.validation.api_key_required")

    if not config.get("model_name", "").strip():
        return False, t("rubric.validation.model_required")

    return True, ""
