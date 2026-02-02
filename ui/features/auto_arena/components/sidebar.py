# -*- coding: utf-8 -*-
"""Sidebar component for Auto Arena feature."""

from typing import Any

import streamlit as st
from shared.constants import DEFAULT_API_ENDPOINTS, DEFAULT_MODELS
from shared.i18n import t

# Session state key for preset sidebar data
STATE_PRESET_SIDEBAR = "arena_preset_sidebar"
STATE_PRESET_LOAD_TRIGGER = "arena_preset_load_trigger"


def _apply_preset_sidebar_data() -> None:
    """Apply preset sidebar data to session state widgets."""
    preset_data = st.session_state.get(STATE_PRESET_SIDEBAR)
    if not preset_data:
        return

    # Judge endpoint - determine provider from URL
    judge_endpoint = preset_data.get("judge_endpoint", "")
    provider = "Custom"
    for prov, url in DEFAULT_API_ENDPOINTS.items():
        if url == judge_endpoint:
            provider = prov
            break

    st.session_state["arena_judge_provider"] = provider
    if provider == "Custom":
        st.session_state["arena_judge_custom_endpoint"] = judge_endpoint

    st.session_state["arena_judge_api_key"] = preset_data.get("judge_api_key", "")

    # Judge model - use stable value for custom option
    CUSTOM_VALUE = "_custom_"
    judge_model = preset_data.get("judge_model", "")
    if judge_model in DEFAULT_MODELS:
        st.session_state["arena_judge_model_value"] = judge_model
    else:
        st.session_state["arena_judge_model_value"] = CUSTOM_VALUE
        st.session_state["arena_judge_custom_model"] = judge_model

    # Evaluation settings
    st.session_state["arena_num_queries"] = preset_data.get("num_queries", 20)
    st.session_state["arena_max_concurrency"] = preset_data.get("max_concurrency", 10)

    # Output settings
    st.session_state["arena_save_queries"] = preset_data.get("save_queries", True)
    st.session_state["arena_save_responses"] = preset_data.get("save_responses", True)
    st.session_state["arena_save_details"] = preset_data.get("save_details", True)
    st.session_state["arena_generate_report"] = preset_data.get("generate_report", True)
    st.session_state["arena_generate_chart"] = preset_data.get("generate_chart", True)

    # Clear the preset data so it doesn't re-apply
    del st.session_state[STATE_PRESET_SIDEBAR]


def _render_judge_settings(config: dict[str, Any]) -> None:
    """Render judge model settings section."""
    st.markdown(f'<div class="section-header">{t("arena.sidebar.judge_model")}</div>', unsafe_allow_html=True)

    provider_options = list(DEFAULT_API_ENDPOINTS.keys())
    if "arena_judge_provider" not in st.session_state:
        st.session_state["arena_judge_provider"] = provider_options[0]

    endpoint_choice = st.selectbox(
        t("api.provider"),
        options=provider_options,
        help=t("arena.sidebar.judge_provider_help"),
        key="arena_judge_provider",
    )

    if endpoint_choice == "Custom":
        api_endpoint = st.text_input(
            t("api.custom_endpoint"),
            placeholder=t("api.custom_endpoint_placeholder"),
            help=t("api.custom_endpoint_help"),
            key="arena_judge_custom_endpoint",
        )
    else:
        api_endpoint = DEFAULT_API_ENDPOINTS[endpoint_choice]

    api_key = st.text_input(
        t("api.key"),
        type="password",
        placeholder=t("api.key_placeholder"),
        help=t("arena.sidebar.judge_api_key_help"),
        key="arena_judge_api_key",
    )

    if api_key:
        st.success(t("api.key_configured"))
    else:
        st.warning(t("api.key_required"))

    # Use stable value for custom option to survive UI language switch
    CUSTOM_VALUE = "_custom_"
    model_options = DEFAULT_MODELS + [CUSTOM_VALUE]

    def format_model_option(x: str) -> str:
        return t("model.custom") if x == CUSTOM_VALUE else x

    # Initialize default value in session state if not exists
    if "arena_judge_model_value" not in st.session_state:
        st.session_state["arena_judge_model_value"] = DEFAULT_MODELS[0] if DEFAULT_MODELS else CUSTOM_VALUE

    model_option = st.selectbox(
        t("model.select"),
        options=model_options,
        format_func=format_model_option,
        key="arena_judge_model_value",
    )

    if model_option == CUSTOM_VALUE:
        model_name = st.text_input(
            t("model.custom_input"),
            placeholder=t("model.custom_placeholder"),
            key="arena_judge_custom_model",
        )
    else:
        model_name = model_option

    config["judge_endpoint"] = api_endpoint
    config["judge_api_key"] = api_key
    config["judge_model"] = model_name


def _render_evaluation_settings(config: dict[str, Any]) -> None:
    """Render evaluation settings section."""
    st.markdown(f'<div class="section-header">{t("arena.sidebar.eval_settings")}</div>', unsafe_allow_html=True)

    # Initialize default values in session state if not exists
    if "arena_num_queries" not in st.session_state:
        st.session_state["arena_num_queries"] = 20
    if "arena_max_concurrency" not in st.session_state:
        st.session_state["arena_max_concurrency"] = 10

    col1, col2 = st.columns(2)

    with col1:
        num_queries = st.number_input(
            t("arena.sidebar.queries"),
            min_value=5,
            max_value=100,
            step=5,
            help=t("arena.sidebar.queries_help"),
            key="arena_num_queries",
        )

    with col2:
        max_concurrency = st.number_input(
            t("arena.sidebar.concurrency"),
            min_value=1,
            help=t("arena.sidebar.concurrency_help"),
            key="arena_max_concurrency",
        )

    config["num_queries"] = num_queries
    config["max_concurrency"] = max_concurrency


def _render_output_settings(config: dict[str, Any]) -> None:
    """Render output settings section."""
    # Initialize default values in session state if not exists
    for key in [
        "arena_save_queries",
        "arena_save_responses",
        "arena_save_details",
        "arena_generate_report",
        "arena_generate_chart",
    ]:
        if key not in st.session_state:
            st.session_state[key] = True

    with st.expander(t("arena.sidebar.output_settings"), expanded=False):
        save_queries = st.checkbox(
            t("arena.sidebar.save_queries"),
            key="arena_save_queries",
        )
        save_responses = st.checkbox(
            t("arena.sidebar.save_responses"),
            key="arena_save_responses",
        )
        save_details = st.checkbox(
            t("arena.sidebar.save_details"),
            key="arena_save_details",
        )
        generate_report = st.checkbox(
            t("arena.sidebar.generate_report"),
            key="arena_generate_report",
        )
        generate_chart = st.checkbox(
            t("arena.sidebar.generate_chart"),
            key="arena_generate_chart",
        )

        config["save_queries"] = save_queries
        config["save_responses"] = save_responses
        config["save_details"] = save_details
        config["generate_report"] = generate_report
        config["generate_chart"] = generate_chart


def render_arena_sidebar() -> dict[str, Any]:
    """Render the Auto Arena sidebar and return configuration.

    Returns:
        Dictionary containing all sidebar configuration values
    """
    # Apply preset sidebar data if available (before rendering widgets)
    if STATE_PRESET_SIDEBAR in st.session_state:
        _apply_preset_sidebar_data()

    config: dict[str, Any] = {}

    _render_judge_settings(config)
    _render_evaluation_settings(config)
    _render_output_settings(config)

    return config
