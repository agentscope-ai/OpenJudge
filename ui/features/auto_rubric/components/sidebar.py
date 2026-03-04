# -*- coding: utf-8 -*-
"""Sidebar component for Auto Rubric feature.

Provides configuration for:
- LLM API settings (provider, key, model)
- Generation settings (language, evaluation mode, score range)
- Advanced settings (max retries)
"""

from typing import Any

import streamlit as st
from shared.constants import DEFAULT_API_ENDPOINTS, DEFAULT_MODELS
from shared.i18n import t


def _render_llm_config(config: dict[str, Any]) -> None:
    """Render LLM configuration section."""
    st.markdown(
        f'<div class="section-header">{t("rubric.sidebar.llm_config")}</div>',
        unsafe_allow_html=True,
    )

    # API Provider selection
    provider_options = list(DEFAULT_API_ENDPOINTS.keys())
    if "rubric_api_provider" not in st.session_state:
        st.session_state["rubric_api_provider"] = provider_options[0]

    provider = st.selectbox(
        t("api.provider"),
        options=provider_options,
        help=t("rubric.sidebar.provider_help"),
        key="rubric_api_provider",
    )

    # Custom endpoint input (only for Custom provider)
    if provider == "Custom":
        api_endpoint = st.text_input(
            t("api.custom_endpoint"),
            placeholder=t("api.custom_endpoint_placeholder"),
            help=t("api.custom_endpoint_help"),
            key="rubric_custom_endpoint",
        )
    else:
        api_endpoint = DEFAULT_API_ENDPOINTS[provider]

    # API Key input
    api_key = st.text_input(
        t("api.key"),
        type="password",
        placeholder=t("api.key_placeholder"),
        help=t("rubric.sidebar.api_key_help"),
        key="rubric_api_key",
    )

    if api_key:
        st.success(t("api.key_configured"))
    else:
        st.warning(t("api.key_required"))

    # Model selection - use stable value for custom option to survive UI language switch
    CUSTOM_VALUE = "_custom_"
    model_options = DEFAULT_MODELS + [CUSTOM_VALUE]

    def format_model_option(x: str) -> str:
        return t("model.custom") if x == CUSTOM_VALUE else x

    # Initialize default value in session state if not exists
    if "rubric_model_value" not in st.session_state:
        st.session_state["rubric_model_value"] = DEFAULT_MODELS[0] if DEFAULT_MODELS else CUSTOM_VALUE

    model_option = st.selectbox(
        t("model.select"),
        options=model_options,
        format_func=format_model_option,
        help=t("rubric.sidebar.model_help"),
        key="rubric_model_value",
    )

    if model_option == CUSTOM_VALUE:
        model_name = st.text_input(
            t("model.custom_input"),
            placeholder=t("model.custom_placeholder"),
            key="rubric_custom_model",
        )
    else:
        model_name = model_option

    config["api_endpoint"] = api_endpoint
    config["api_key"] = api_key
    config["model_name"] = model_name


def _render_generation_settings(config: dict[str, Any]) -> None:
    """Render generation settings section."""
    st.markdown(
        f'<div class="section-header">{t("rubric.sidebar.gen_settings")}</div>',
        unsafe_allow_html=True,
    )

    # Language selection - use stable keys to survive UI language switch
    language_values = ["EN", "ZH"]
    language_labels = {"EN": "English", "ZH": "中文"}

    # Initialize default value in session state if not exists
    if "rubric_language_value" not in st.session_state:
        st.session_state["rubric_language_value"] = "EN"

    language = st.selectbox(
        t("rubric.sidebar.language"),
        options=language_values,
        format_func=lambda x: language_labels.get(x, x),
        help=t("rubric.sidebar.language_help"),
        key="rubric_language_value",
    )

    # Evaluation mode - use stable values to survive UI language switch
    mode_values = ["pointwise", "listwise"]
    mode_labels = {
        "pointwise": t("rubric.sidebar.pointwise"),
        "listwise": t("rubric.sidebar.listwise"),
    }

    # Initialize default value in session state if not exists
    if "rubric_eval_mode_value" not in st.session_state:
        st.session_state["rubric_eval_mode_value"] = "pointwise"

    grader_mode = st.selectbox(
        t("rubric.sidebar.eval_mode"),
        options=mode_values,
        format_func=lambda x: mode_labels.get(x, x),
        help=t("rubric.sidebar.eval_mode_help"),
        key="rubric_eval_mode_value",
    )

    # Note: Score range for pointwise mode is now automatically extracted from uploaded data
    # No manual input needed here

    config["language"] = language
    config["grader_mode"] = grader_mode


def _render_advanced_settings(config: dict[str, Any]) -> None:
    """Render advanced settings section."""
    with st.expander(t("rubric.sidebar.advanced"), expanded=False):
        max_retries = st.number_input(
            t("rubric.sidebar.max_retries"),
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help=t("rubric.sidebar.max_retries_help"),
            key="rubric_max_retries",
        )
        config["max_retries"] = max_retries


def render_rubric_sidebar() -> dict[str, Any]:
    """Render the Auto Rubric sidebar and return configuration.

    Returns:
        Dictionary containing all sidebar configuration values:
        - api_endpoint: API endpoint URL
        - api_key: API key for authentication
        - model_name: Model name to use
        - language: Language for generation (EN/ZH)
        - grader_mode: Evaluation mode (pointwise/listwise)
        - min_score: Minimum score (pointwise only)
        - max_score: Maximum score (pointwise only)
        - max_retries: Maximum retry attempts
    """
    config: dict[str, Any] = {}

    _render_llm_config(config)
    _render_generation_settings(config)
    _render_advanced_settings(config)

    return config
