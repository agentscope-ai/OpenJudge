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
        load_example = st.button("Load Example", use_container_width=True)
    with col_btn2:
        clear_all = st.button("Clear All", use_container_width=True)

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
    tab_main, tab_tools, tab_context = st.tabs(["Query", "Tools", "Context"])

    with tab_main:
        query = st.text_area(
            "Query",
            value=defaults.get("query", ""),
            height=100,
            placeholder="Enter the user's query to the agent...",
            help="The task or question given to the agent",
            key=f"{form_key}_query",
        )
        input_data["query"] = query

    with tab_tools:
        st.markdown(
            """<div class="info-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    Enter tool definitions and calls in JSON format
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        tool_definitions = st.text_area(
            "Available Tool Definitions (JSON)",
            value=defaults.get("tool_definitions", ""),
            height=200,
            placeholder='[{"name": "get_weather", "description": "...", "parameters": {...}}]',
            help="JSON array of available tool definitions",
            key=f"{form_key}_tool_defs",
        )
        input_data["tool_definitions"] = tool_definitions

        tool_calls = st.text_area(
            "Agent's Tool Calls (JSON)",
            value=defaults.get("tool_calls", ""),
            height=150,
            placeholder='[{"name": "get_weather", "arguments": {"location": "Beijing"}}]',
            help="JSON array of tool calls made by the agent",
            key=f"{form_key}_tool_calls",
        )
        input_data["tool_calls"] = tool_calls

        if "reference_tool_calls" in input_fields:
            reference_tool_calls = st.text_area(
                "Expected Tool Calls (JSON)",
                value="",
                height=150,
                placeholder='[{"name": "get_weather", "arguments": {"location": "Beijing"}}]',
                help="JSON array of expected/correct tool calls",
                key=f"{form_key}_ref_tool_calls",
            )
            input_data["reference_tool_calls"] = reference_tool_calls

    with tab_context:
        context = st.text_area(
            "Additional Context",
            value="",
            height=200,
            placeholder="Enter any additional context...",
            help="Optional background information",
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
    tab_main, tab_context = st.tabs(["Main Input", "Context"])

    with tab_main:
        if "query" in input_fields:
            query = st.text_area(
                "Query",
                value=defaults.get("query", ""),
                height=100,
                placeholder="Enter the user's question or prompt...",
                help="The original question or prompt from the user",
                key=f"{form_key}_query",
            )
            input_data["query"] = query

        response = st.text_area(
            "Response to Evaluate",
            value=defaults.get("response", ""),
            height=150,
            placeholder="Enter the response to be evaluated...",
            help="The model's response that needs to be evaluated",
            key=f"{form_key}_response",
        )
        input_data["response"] = response

        requires_reference = grader_config.get("requires_reference", False)
        if "reference_response" in input_fields or requires_reference:
            ref_label = "Reference Response *" if requires_reference else "Reference Response (Optional)"
            placeholder_suffix = " (Required)" if requires_reference else ""
            reference_response = st.text_area(
                ref_label,
                value=defaults.get("reference_response", ""),
                height=120,
                placeholder=f"Enter the reference/golden answer...{placeholder_suffix}",
                help="The expected or ideal response for comparison",
                key=f"{form_key}_ref_response",
            )
            input_data["reference_response"] = reference_response

    with tab_context:
        context = st.text_area(
            "Additional Context",
            value=defaults.get("context", ""),
            height=200,
            placeholder="Enter any additional context that might help with evaluation...",
            help="Optional background information for the evaluation",
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

    render_section_header("Input Data")

    # Action buttons (outside form for immediate effect)
    defaults = _render_action_buttons(category)
    input_data: dict[str, Any] = {}
    run_flag = False

    if not grader_config:
        st.warning("Please select a grader from the sidebar")
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
            "Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

        if not can_run:
            missing = []
            if requires_model and not api_key:
                missing.append("API Key")
            if requires_model and not model_name:
                missing.append("Model")
            if not grader_name:
                missing.append("Grader")
            if not has_content:
                missing.append("Required input data")
            if missing:
                st.caption(f"Missing: {', '.join(missing)}")

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
        "Run Evaluation",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if not can_run:
        missing = []
        if requires_model and not api_key:
            missing.append("API Key")
        if requires_model and not model_name:
            missing.append("Model")
        if not grader_name:
            missing.append("Grader")
        if not has_content:
            missing.append("Required input data")
        if missing:
            st.caption(f"Missing: {', '.join(missing)}")

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

    render_section_header("Input Data")

    # Action buttons and get defaults
    defaults = _render_action_buttons(category)
    input_data: dict[str, Any] = {}

    if not grader_config:
        st.warning("Please select a grader from the sidebar")
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
