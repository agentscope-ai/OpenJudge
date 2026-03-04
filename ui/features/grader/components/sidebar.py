# -*- coding: utf-8 -*-
"""Sidebar component for Grader feature."""

from typing import Any

import streamlit as st
from features.grader.config.constants import (
    DEFAULT_API_ENDPOINTS,
    DEFAULT_MODELS,
    GRADER_CATEGORIES,
)
from features.grader.config.grader_registry import (
    GRADER_REGISTRY,
    get_graders_by_category,
)
from shared.i18n import get_ui_language, t

from openjudge.models.schema.prompt_template import LanguageEnum


def _render_api_settings(config: dict[str, Any]) -> None:
    """Render API settings section."""
    st.markdown(f'<div class="section-header">{t("api.settings")}</div>', unsafe_allow_html=True)

    provider_options = list(DEFAULT_API_ENDPOINTS.keys())
    if "grader_api_provider" not in st.session_state:
        st.session_state["grader_api_provider"] = provider_options[0]

    endpoint_choice = st.selectbox(
        t("api.provider"),
        options=provider_options,
        help=t("api.provider_help"),
        key="grader_api_provider",
    )

    if endpoint_choice == "Custom":
        api_endpoint = st.text_input(
            t("api.custom_endpoint"),
            placeholder=t("api.custom_endpoint_placeholder"),
            help=t("api.custom_endpoint_help"),
            key="grader_custom_endpoint",
        )
    else:
        api_endpoint = DEFAULT_API_ENDPOINTS[endpoint_choice]

    api_key = st.text_input(
        t("api.key"),
        type="password",
        placeholder=t("api.key_placeholder"),
        help=t("api.key_help"),
        key="grader_api_key",
    )

    if api_key:
        st.success(t("api.key_configured"))
    else:
        st.warning(t("api.key_required"))

    config["api_endpoint"] = api_endpoint
    config["api_key"] = api_key


def _render_model_settings(config: dict[str, Any]) -> None:
    """Render model settings section."""
    st.markdown(f'<div class="section-header">{t("model.settings")}</div>', unsafe_allow_html=True)

    # Use stable value "_custom_" for custom option to survive UI language switch
    CUSTOM_VALUE = "_custom_"
    model_options = DEFAULT_MODELS + [CUSTOM_VALUE]

    def format_model_option(x: str) -> str:
        return t("model.custom") if x == CUSTOM_VALUE else x

    # Initialize default value in session state if not exists
    if "grader_model_select" not in st.session_state:
        st.session_state["grader_model_select"] = DEFAULT_MODELS[0] if DEFAULT_MODELS else CUSTOM_VALUE

    model_option = st.selectbox(
        t("model.select"),
        options=model_options,
        format_func=format_model_option,
        label_visibility="collapsed",
        key="grader_model_select",
    )

    if model_option == CUSTOM_VALUE:
        model_name = st.text_input(
            t("model.custom_input"),
            placeholder=t("model.custom_placeholder"),
            label_visibility="collapsed",
            key="grader_custom_model",
        )
    else:
        model_name = model_option

    config["model_name"] = model_name


def _render_grader_settings(config: dict[str, Any]) -> None:
    """Render grader settings section."""
    st.markdown(f'<div class="section-header">{t("grader.sidebar.grader")}</div>', unsafe_allow_html=True)

    category_options = list(GRADER_CATEGORIES.keys())
    category_labels = [GRADER_CATEGORIES[cat]["name"] for cat in category_options]

    # Initialize category selection in session state
    if "grader_category_idx" not in st.session_state:
        st.session_state["grader_category_idx"] = 0

    selected_category_idx = st.selectbox(
        t("grader.sidebar.category"),
        options=range(len(category_options)),
        format_func=lambda x: category_labels[x],
        help=t("grader.sidebar.category_help"),
        key="grader_category_idx",
    )
    selected_category = category_options[selected_category_idx]
    config["grader_category"] = selected_category

    graders_in_category = get_graders_by_category(selected_category)
    grader_names = list(graders_in_category.keys())

    if grader_names:
        # Initialize grader selection in session state
        # Reset grader selection when category changes
        grader_key = f"grader_selection_{selected_category}"
        if grader_key not in st.session_state:
            st.session_state[grader_key] = 0

        selected_grader_idx = st.selectbox(
            t("grader.sidebar.grader"),
            options=range(len(grader_names)),
            format_func=lambda x: grader_names[x],
            label_visibility="collapsed",
            key=grader_key,
        )
        selected_grader = grader_names[selected_grader_idx]
        grader_config = GRADER_REGISTRY[selected_grader]

        # Get localized name and description based on UI language
        is_chinese = get_ui_language() == "zh"
        display_name = grader_config.get("name_zh", selected_grader) if is_chinese else selected_grader
        display_desc = (
            grader_config.get("description_zh", grader_config.get("description", ""))
            if is_chinese
            else grader_config.get("description", "")
        )

        st.markdown(
            f"""<div class="info-card">
                <div style="font-weight: 500; color: #F1F5F9; margin-bottom: 0.25rem;">
                    {display_name}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    {display_desc}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        config["grader_name"] = selected_grader
        config["grader_config"] = grader_config
    else:
        st.warning(t("grader.sidebar.no_graders"))
        config["grader_name"] = None
        config["grader_config"] = None


def _render_evaluation_settings(config: dict[str, Any]) -> None:
    """Render evaluation settings section."""
    st.markdown(f'<div class="section-header">{t("grader.sidebar.settings")}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Use stable values to survive UI language switch
        language_values = ["EN", "ZH"]
        language_labels = {"EN": "English", "ZH": "中文"}

        # Initialize default value in session state if not exists
        if "grader_language_select" not in st.session_state:
            st.session_state["grader_language_select"] = "EN"

        language_option = st.selectbox(
            t("grader.sidebar.language"),
            options=language_values,
            format_func=lambda x: language_labels.get(x, x),
            help=t("grader.sidebar.language_help"),
            key="grader_language_select",
        )
        config["language"] = LanguageEnum.EN if language_option == "EN" else LanguageEnum.ZH

    with col2:
        if config.get("grader_config"):
            score_range = config["grader_config"].get("score_range", (0, 1))
            default_threshold = config["grader_config"].get("default_threshold", 0.5)

            if score_range == (1, 5):
                # Initialize threshold in session state if not exists
                if "grader_threshold_select" not in st.session_state:
                    st.session_state["grader_threshold_select"] = int(default_threshold)

                threshold = st.selectbox(
                    t("grader.sidebar.threshold"),
                    options=[1, 2, 3, 4, 5],
                    help=t("grader.sidebar.threshold_help"),
                    key="grader_threshold_select",
                )
            else:
                # Initialize threshold slider in session state if not exists
                if "grader_threshold_slider" not in st.session_state:
                    st.session_state["grader_threshold_slider"] = float(default_threshold)

                threshold = st.slider(
                    t("grader.sidebar.threshold"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    help=t("grader.sidebar.threshold_help"),
                    key="grader_threshold_slider",
                )
            config["threshold"] = threshold
        else:
            config["threshold"] = 0.5


def _render_extra_params(config: dict[str, Any]) -> None:
    """Render extra parameters section."""
    if not config.get("grader_config") or not config["grader_config"].get("extra_params"):
        config["extra_params"] = {}
        return

    with st.expander(t("grader.sidebar.advanced_options")):
        extra_params = config["grader_config"]["extra_params"]
        config["extra_params"] = {}

        for param_name, param_config in extra_params.items():
            param_type = param_config.get("type", "text")
            default_value = param_config.get("default")
            description = param_config.get("description", "")
            label = param_name.replace("_", " ").title()

            if param_type == "select":
                options = param_config.get("options", [])
                idx = options.index(default_value) if default_value in options else 0
                value = st.selectbox(label, options=options, index=idx, help=description)
            elif param_type == "checkbox":
                value = st.checkbox(label, value=default_value, help=description)
            elif param_type == "number":
                value = st.number_input(label, value=default_value, help=description)
            else:
                value = st.text_input(
                    label,
                    value=str(default_value) if default_value else "",
                    help=description,
                )

            config["extra_params"][param_name] = value


def _render_vision_warning(config: dict[str, Any]) -> None:
    """Render vision model warning if needed."""
    if config.get("grader_config") and config["grader_config"].get("requires_vision_model"):
        st.info(t("grader.sidebar.vision_warning"))


def _render_batch_settings(config: dict[str, Any]) -> None:
    """Render batch evaluation settings section."""
    st.markdown(f'<div class="section-header">{t("grader.sidebar.batch_settings")}</div>', unsafe_allow_html=True)

    max_concurrency = st.number_input(
        t("grader.sidebar.max_concurrency"),
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help=t("grader.sidebar.max_concurrency_help"),
    )
    config["max_concurrency"] = max_concurrency


def render_grader_sidebar(batch_mode: bool = False) -> dict[str, Any]:
    """Render the grader sidebar and return configuration.

    Args:
        batch_mode: Whether to show batch evaluation settings

    Returns:
        Dictionary containing all sidebar configuration values
    """
    config: dict[str, Any] = {}

    _render_api_settings(config)
    _render_model_settings(config)
    _render_grader_settings(config)
    _render_evaluation_settings(config)
    _render_extra_params(config)

    if batch_mode:
        _render_batch_settings(config)

    _render_vision_warning(config)

    return config
