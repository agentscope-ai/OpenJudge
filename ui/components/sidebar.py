# -*- coding: utf-8 -*-
"""Sidebar component for OpenJudge Studio."""

from pathlib import Path
from typing import Any

import streamlit as st
from config.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_API_ENDPOINTS,
    DEFAULT_MODELS,
    GRADER_CATEGORIES,
)
from config.grader_registry import GRADER_REGISTRY, get_graders_by_category

from openjudge.models.schema.prompt_template import LanguageEnum

LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.svg"


def _render_logo_and_title() -> None:
    """Render logo and title section."""
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
        else:
            st.markdown(
                """<div style="width: 48px; height: 48px; margin: 0 auto;
                    background: linear-gradient(135deg, #6366F1, #8B5CF6);
                    border-radius: 12px; display: flex; align-items: center;
                    justify-content: center; font-size: 1.5rem; color: white;
                    font-weight: 700;">OJ</div>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""<div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F1F5F9;">
                {APP_NAME}
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">
                v{APP_VERSION} • Powered by OpenJudge
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def _render_api_settings(config: dict[str, Any]) -> None:
    """Render API settings section."""
    st.markdown('<div class="section-header">API Settings</div>', unsafe_allow_html=True)

    endpoint_choice = st.selectbox(
        "Provider",
        options=list(DEFAULT_API_ENDPOINTS.keys()),
        index=0,
        help="Select API provider or choose Custom",
    )

    if endpoint_choice == "Custom":
        api_endpoint = st.text_input(
            "Custom Endpoint",
            placeholder="https://api.example.com/v1",
            help="Enter your custom API endpoint URL",
        )
    else:
        api_endpoint = DEFAULT_API_ENDPOINTS[endpoint_choice]

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Your API key for the LLM service",
    )

    if api_key:
        st.success("API Key configured")
    else:
        st.warning("Enter API Key to continue")

    config["api_endpoint"] = api_endpoint
    config["api_key"] = api_key
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def _render_model_settings(config: dict[str, Any]) -> None:
    """Render model settings section."""
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)

    model_option = st.selectbox(
        "Model",
        options=DEFAULT_MODELS + ["Custom..."],
        index=0,
        label_visibility="collapsed",
    )

    if model_option == "Custom...":
        model_name = st.text_input(
            "Custom Model",
            placeholder="Enter model name",
            label_visibility="collapsed",
        )
    else:
        model_name = model_option

    config["model_name"] = model_name
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def _render_grader_settings(config: dict[str, Any]) -> None:
    """Render grader settings section."""
    st.markdown('<div class="section-header">Grader</div>', unsafe_allow_html=True)

    category_options = list(GRADER_CATEGORIES.keys())
    category_labels = [GRADER_CATEGORIES[cat]["name"] for cat in category_options]

    selected_category_idx = st.selectbox(
        "Category",
        options=range(len(category_options)),
        format_func=lambda x: category_labels[x],
        index=0,
        help="Select grader category",
    )
    selected_category = category_options[selected_category_idx]
    config["grader_category"] = selected_category

    graders_in_category = get_graders_by_category(selected_category)
    grader_names = list(graders_in_category.keys())

    if grader_names:
        selected_grader_idx = st.selectbox(
            "Grader",
            options=range(len(grader_names)),
            format_func=lambda x: grader_names[x],
            index=0,
            label_visibility="collapsed",
        )
        selected_grader = grader_names[selected_grader_idx]
        grader_config = GRADER_REGISTRY[selected_grader]

        st.markdown(
            f"""<div class="info-card">
                <div style="font-weight: 500; color: #F1F5F9; margin-bottom: 0.25rem;">
                    {grader_config['name_zh']}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    {grader_config['description_zh']}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        config["grader_name"] = selected_grader
        config["grader_config"] = grader_config
    else:
        st.warning("No graders in this category")
        config["grader_name"] = None
        config["grader_config"] = None

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def _render_evaluation_settings(config: dict[str, Any]) -> None:
    """Render evaluation settings section."""
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        language_option = st.selectbox(
            "Language",
            options=["EN", "中文"],
            index=0,
            help="Evaluation language",
        )
        config["language"] = LanguageEnum.EN if language_option == "EN" else LanguageEnum.ZH

    with col2:
        if config.get("grader_config"):
            score_range = config["grader_config"].get("score_range", (0, 1))
            default_threshold = config["grader_config"].get("default_threshold", 0.5)

            if score_range == (1, 5):
                threshold = st.selectbox(
                    "Threshold",
                    options=[1, 2, 3, 4, 5],
                    index=int(default_threshold) - 1,
                    help="Pass threshold",
                )
            else:
                threshold = st.slider(
                    "Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_threshold),
                    step=0.1,
                    help="Pass threshold",
                )
            config["threshold"] = threshold
        else:
            config["threshold"] = 0.5


def _render_extra_params(config: dict[str, Any]) -> None:
    """Render extra parameters section."""
    if not config.get("grader_config") or not config["grader_config"].get("extra_params"):
        config["extra_params"] = {}
        return

    with st.expander("Advanced Options"):
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
        st.info("This grader requires a vision-capable model (e.g., qwen-vl-max, gpt-4o)")


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar and return configuration.

    Returns:
        Dictionary containing all sidebar configuration values
    """
    config: dict[str, Any] = {}

    with st.sidebar:
        _render_logo_and_title()
        _render_api_settings(config)
        _render_model_settings(config)
        _render_grader_settings(config)
        _render_evaluation_settings(config)
        _render_extra_params(config)
        _render_vision_warning(config)

    return config
