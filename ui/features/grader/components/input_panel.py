# -*- coding: utf-8 -*-
"""Input panel component for Grader feature."""

from typing import Any

import streamlit as st
from features.grader.components.multimodal import (
    render_multimodal_input,
    render_text_to_image_input,
)
from features.grader.config.constants import EXAMPLE_DATA
from shared.components.common import render_section_header
from shared.i18n import t


def _get_example_data(category: str) -> dict[str, Any]:
    """Get appropriate example data based on grader category.

    Args:
        category: Grader category

    Returns:
        Example data dictionary
    """
    category_map = {
        "text": "text_similarity",
        "code": "code_style",
        "math": "math_verify",
        "multimodal": "multimodal",
        "agent": "agent_tool",
    }
    example_key = category_map.get(category, "default")
    return EXAMPLE_DATA.get(example_key, EXAMPLE_DATA["default"])


def _render_action_buttons(category: str) -> dict[str, Any]:
    """Render action buttons and return default values.

    Args:
        category: Grader category

    Returns:
        Default values dictionary
    """
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        load_example = st.button(t("grader.input.load_example"), use_container_width=True)
    with col_btn2:
        clear_all = st.button(t("grader.input.clear_all"), use_container_width=True)

    if load_example:
        st.session_state.example_loaded = True
        st.session_state.evaluation_result = None
    if clear_all:
        st.session_state.example_loaded = False
        st.session_state.evaluation_result = None

    if st.session_state.get("example_loaded", False):
        return _get_example_data(category)
    return {
        "query": "",
        "response": "",
        "reference_response": "",
        "context": "",
        "tool_definitions": "",
        "tool_calls": "",
    }


def _render_agent_input(defaults: dict[str, Any], input_fields: list, form_key: str) -> dict[str, Any]:
    """Render agent grader input fields.

    Args:
        defaults: Default values
        input_fields: List of input fields
        form_key: Unique form key

    Returns:
        Input data dictionary
    """
    input_data: dict[str, Any] = {}
    tab_main, tab_tools, tab_context = st.tabs(
        [
            t("grader.input.tab_query"),
            t("grader.input.tab_tools"),
            t("grader.input.tab_context"),
        ]
    )

    with tab_main:
        query = st.text_area(
            t("grader.input.query"),
            value=defaults.get("query", ""),
            height=100,
            placeholder=t("grader.input.agent_query_placeholder"),
            help=t("grader.input.agent_query_help"),
            key=f"{form_key}_query",
        )
        input_data["query"] = query

    with tab_tools:
        st.markdown(
            f"""<div class="info-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    {t("grader.input.tool_definitions_info")}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        tool_definitions = st.text_area(
            t("grader.input.tool_definitions"),
            value=defaults.get("tool_definitions", ""),
            height=200,
            placeholder=t("grader.input.tool_definitions_placeholder"),
            help=t("grader.input.tool_definitions_help"),
            key=f"{form_key}_tool_defs",
        )
        input_data["tool_definitions"] = tool_definitions

        tool_calls = st.text_area(
            t("grader.input.tool_calls"),
            value=defaults.get("tool_calls", ""),
            height=150,
            placeholder=t("grader.input.tool_calls_placeholder"),
            help=t("grader.input.tool_calls_help"),
            key=f"{form_key}_tool_calls",
        )
        input_data["tool_calls"] = tool_calls

        if "reference_tool_calls" in input_fields:
            reference_tool_calls = st.text_area(
                t("grader.input.reference_tool_calls"),
                value="",
                height=150,
                placeholder=t("grader.input.tool_calls_placeholder"),
                help=t("grader.input.reference_tool_calls_help"),
                key=f"{form_key}_ref_tool_calls",
            )
            input_data["reference_tool_calls"] = reference_tool_calls

    with tab_context:
        context = st.text_area(
            t("grader.input.context"),
            value="",
            height=200,
            placeholder=t("grader.input.context_placeholder"),
            help=t("grader.input.context_help"),
            key=f"{form_key}_context",
        )
        input_data["context"] = context

    input_data["has_content"] = bool(query and tool_definitions and tool_calls)
    return input_data


def _render_multimodal_inputs(input_fields: list[str]) -> dict[str, Any]:
    """Render multimodal grader input fields.

    Args:
        input_fields: List of input fields

    Returns:
        Input data dictionary
    """
    input_data: dict[str, Any] = {}

    if "response_multimodal" in input_fields:
        content_list, _ = render_multimodal_input()
        input_data["response"] = content_list
        input_data["has_content"] = len(content_list) > 0
    elif "response_image" in input_fields:
        text_prompt, image = render_text_to_image_input()
        input_data["query"] = text_prompt
        input_data["response"] = image
        input_data["has_content"] = bool(text_prompt and image)

    return input_data


def _render_standard_input(
    defaults: dict[str, Any],
    input_fields: list,
    grader_config: dict[str, Any],
    form_key: str,
) -> dict[str, Any]:
    """Render standard grader input fields.

    Args:
        defaults: Default values
        input_fields: List of input fields
        grader_config: Grader configuration
        form_key: Unique form key

    Returns:
        Input data dictionary
    """
    input_data: dict[str, Any] = {}
    tab_main, tab_context = st.tabs([t("grader.input.tab_main"), t("grader.input.tab_context")])

    with tab_main:
        if "query" in input_fields:
            query = st.text_area(
                t("grader.input.query"),
                value=defaults.get("query", ""),
                height=100,
                placeholder=t("grader.input.query_placeholder"),
                help=t("grader.input.query_help"),
                key=f"{form_key}_query",
            )
            input_data["query"] = query

        response = st.text_area(
            t("grader.input.response"),
            value=defaults.get("response", ""),
            height=150,
            placeholder=t("grader.input.response_placeholder"),
            help=t("grader.input.response_help"),
            key=f"{form_key}_response",
        )
        input_data["response"] = response

        requires_reference = grader_config.get("requires_reference", False)
        if "reference_response" in input_fields or requires_reference:
            ref_label = (
                t("grader.input.reference_required") if requires_reference else t("grader.input.reference_optional")
            )
            reference_response = st.text_area(
                ref_label,
                value=defaults.get("reference_response", ""),
                height=120,
                placeholder=t("grader.input.reference_placeholder"),
                help=t("grader.input.reference_help"),
                key=f"{form_key}_ref_response",
            )
            input_data["reference_response"] = reference_response

    with tab_context:
        context = st.text_area(
            t("grader.input.context"),
            value=defaults.get("context", ""),
            height=200,
            placeholder=t("grader.input.context_placeholder"),
            help=t("grader.input.context_help"),
            key=f"{form_key}_context",
        )
        input_data["context"] = context

    # Determine if we have enough content
    has_content = bool(input_data.get("response", ""))
    if "query" in input_fields:
        has_content = has_content and bool(input_data.get("query", ""))
    if grader_config.get("requires_reference", False):
        has_content = has_content and bool(input_data.get("reference_response", ""))
    input_data["has_content"] = has_content

    return input_data


def render_input_panel_with_button(sidebar_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Render the input panel with submit button using st.form.

    Using st.form ensures all input values are submitted together when the button
    is clicked, without requiring users to press Enter or click outside the input fields.

    Args:
        sidebar_config: Configuration from sidebar

    Returns:
        Tuple of (input_data dict, run_flag bool)
    """
    grader_config = sidebar_config.get("grader_config")
    category = sidebar_config.get("grader_category", "common")
    grader_name = sidebar_config.get("grader_name", "default")

    render_section_header(t("grader.input.title"))

    # Action buttons (outside form for immediate effect)
    defaults = _render_action_buttons(category)
    input_data: dict[str, Any] = {}
    run_flag = False

    if not grader_config:
        st.warning(t("grader.input.select_grader_first"))
        return input_data, run_flag

    input_fields = grader_config.get("input_fields", ["query", "response"])

    # Multimodal Graders - no form needed (file upload doesn't work well in forms)
    if "response_multimodal" in input_fields or "response_image" in input_fields:
        input_data = _render_multimodal_inputs(input_fields)
        run_flag = _render_run_button_standalone(sidebar_config, input_data)
        return input_data, run_flag

    # Use form for text-based inputs
    form_key = f"grader_form_{grader_name}"
    with st.form(key=form_key, clear_on_submit=False):
        # Agent Graders
        if "tool_definitions" in input_fields:
            input_data = _render_agent_input(defaults, input_fields, form_key)
        else:
            # Standard Graders
            input_data = _render_standard_input(defaults, input_fields, grader_config, form_key)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Submit button inside form
        api_key = sidebar_config.get("api_key", "")
        model_name = sidebar_config.get("model_name", "")
        has_content = input_data.get("has_content", False)
        requires_model = grader_config.get("requires_model", True) if grader_config else True

        can_run = bool(grader_name and has_content)
        if requires_model:
            can_run = can_run and bool(api_key and model_name)

        run_flag = st.form_submit_button(
            t("grader.input.run"),
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

        if not can_run:
            missing = []
            if requires_model and not api_key:
                missing.append(t("grader.input.missing_api_key"))
            if requires_model and not model_name:
                missing.append(t("grader.input.missing_model"))
            if not grader_name:
                missing.append(t("grader.input.missing_grader"))
            if not has_content:
                missing.append(t("grader.input.missing_data"))
            if missing:
                st.caption(f"{t('grader.input.missing')}: {', '.join(missing)}")

    return input_data, run_flag


def _render_run_button_standalone(
    sidebar_config: dict[str, Any],
    input_data: dict[str, Any],
) -> bool:
    """Render standalone run button (for multimodal inputs).

    Args:
        sidebar_config: Configuration from sidebar
        input_data: Input data from input panel

    Returns:
        True if button was clicked and evaluation should run
    """
    api_key = sidebar_config.get("api_key", "")
    model_name = sidebar_config.get("model_name", "")
    grader_name = sidebar_config.get("grader_name")
    grader_config = sidebar_config.get("grader_config")
    has_content = input_data.get("has_content", False)

    requires_model = grader_config.get("requires_model", True) if grader_config else True

    can_run = bool(grader_name and has_content)
    if requires_model:
        can_run = can_run and bool(api_key and model_name)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    run_button = st.button(
        t("grader.input.run"),
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if not can_run:
        missing = []
        if requires_model and not api_key:
            missing.append(t("grader.input.missing_api_key"))
        if requires_model and not model_name:
            missing.append(t("grader.input.missing_model"))
        if not grader_name:
            missing.append(t("grader.input.missing_grader"))
        if not has_content:
            missing.append(t("grader.input.missing_data"))
        if missing:
            st.caption(f"{t('grader.input.missing')}: {', '.join(missing)}")

    return run_button


# Keep old functions for backward compatibility
def render_input_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the input panel and return input data.

    DEPRECATED: Use render_input_panel_with_button instead.

    Args:
        sidebar_config: Configuration from sidebar

    Returns:
        Dictionary containing all input data
    """
    grader_config = sidebar_config.get("grader_config")
    category = sidebar_config.get("grader_category", "common")
    grader_name = sidebar_config.get("grader_name", "default")

    render_section_header(t("grader.input.title"))

    # Action buttons and get defaults
    defaults = _render_action_buttons(category)
    input_data: dict[str, Any] = {}

    if not grader_config:
        st.warning(t("grader.input.select_grader_first"))
        return input_data

    input_fields = grader_config.get("input_fields", ["query", "response"])
    form_key = f"grader_input_{grader_name}"

    # Multimodal Graders
    if "response_multimodal" in input_fields or "response_image" in input_fields:
        return _render_multimodal_inputs(input_fields)

    # Agent Graders
    if "tool_definitions" in input_fields:
        return _render_agent_input(defaults, input_fields, form_key)

    # Standard Graders
    return _render_standard_input(defaults, input_fields, grader_config, form_key)


def render_run_button(
    sidebar_config: dict[str, Any],
    input_data: dict[str, Any],
) -> bool:
    """Render the run evaluation button.

    DEPRECATED: Use render_input_panel_with_button instead.

    Args:
        sidebar_config: Configuration from sidebar
        input_data: Input data from input panel

    Returns:
        True if button was clicked and evaluation should run
    """
    return _render_run_button_standalone(sidebar_config, input_data)
