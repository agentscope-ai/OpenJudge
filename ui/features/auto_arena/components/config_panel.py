# -*- coding: utf-8 -*-
"""Configuration panel for Auto Arena feature.

Provides UI for configuring:
- Configuration presets (save/load)
- Task description and scenario
- Target endpoints (models/agents to evaluate)
- Query generation settings
"""

from typing import Any

import streamlit as st
from features.auto_arena.components.preset_panel import render_preset_panel
from shared.constants import DEFAULT_API_ENDPOINTS, DEFAULT_MODELS
from shared.i18n import t

# Session state keys for preset loading
STATE_PRESET_LOAD_TRIGGER = "arena_preset_load_trigger"


def _apply_preset_to_session_state(ui_state: dict[str, Any]) -> None:
    """Apply loaded preset data to session state.

    This updates the widget keys so forms display the loaded values.

    Args:
        ui_state: UI state dictionary from PresetManager.config_to_ui_state()
    """
    # Task configuration
    st.session_state["arena_task_description"] = ui_state.get("task_description", "")
    st.session_state["arena_task_scenario"] = ui_state.get("task_scenario", "")

    # Query settings
    seed_queries = ui_state.get("seed_queries", [])
    st.session_state["arena_seed_queries"] = "\n".join(seed_queries) if seed_queries else ""
    st.session_state["arena_query_temp"] = ui_state.get("query_temperature", 0.9)
    st.session_state["arena_max_similarity"] = ui_state.get("max_similarity", 0.85)
    st.session_state["arena_enable_evolution"] = ui_state.get("enable_evolution", False)

    # Target endpoints - store in a special key for reconstruction
    st.session_state["arena_preset_endpoints"] = ui_state.get("target_endpoints", [])

    # Sidebar settings - store for sidebar to pick up
    st.session_state["arena_preset_sidebar"] = {
        "judge_endpoint": ui_state.get("judge_endpoint", ""),
        "judge_api_key": ui_state.get("judge_api_key", ""),
        "judge_model": ui_state.get("judge_model", ""),
        "num_queries": ui_state.get("num_queries", 20),
        "max_concurrency": ui_state.get("max_concurrency", 10),
        "save_queries": ui_state.get("save_queries", True),
        "save_responses": ui_state.get("save_responses", True),
        "save_details": ui_state.get("save_details", True),
        "generate_report": ui_state.get("generate_report", True),
        "generate_chart": ui_state.get("generate_chart", True),
    }

    # Mark that preset was loaded (sidebar will check this)
    st.session_state[STATE_PRESET_LOAD_TRIGGER] = True


def _on_preset_load(ui_state: dict[str, Any]) -> None:
    """Callback when a preset is loaded.

    Args:
        ui_state: UI state dictionary from preset
    """
    _apply_preset_to_session_state(ui_state)


def _get_current_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Get current configuration from session state.

    Returns:
        Tuple of (sidebar_config, panel_config)
    """
    # Collect sidebar config from session state
    sidebar_config = {
        "judge_endpoint": st.session_state.get("arena_judge_custom_endpoint", "")
        or DEFAULT_API_ENDPOINTS.get(st.session_state.get("arena_judge_provider", "DashScope"), ""),
        "judge_api_key": st.session_state.get("arena_judge_api_key", ""),
        "judge_model": st.session_state.get("arena_judge_custom_model", "")
        or st.session_state.get("arena_judge_model", ""),
        "num_queries": st.session_state.get("arena_num_queries", 20),
        "max_concurrency": st.session_state.get("arena_max_concurrency", 10),
        "save_queries": st.session_state.get("arena_save_queries", True),
        "save_responses": st.session_state.get("arena_save_responses", True),
        "save_details": st.session_state.get("arena_save_details", True),
        "generate_report": st.session_state.get("arena_generate_report", True),
        "generate_chart": st.session_state.get("arena_generate_chart", True),
    }

    # Collect panel config
    panel_config = {
        "task_description": st.session_state.get("arena_task_description", ""),
        "task_scenario": st.session_state.get("arena_task_scenario", ""),
        "seed_queries": [q.strip() for q in st.session_state.get("arena_seed_queries", "").split("\n") if q.strip()],
        "query_temperature": st.session_state.get("arena_query_temp", 0.9),
        "max_similarity": st.session_state.get("arena_max_similarity", 0.85),
        "enable_evolution": st.session_state.get("arena_enable_evolution", False),
        "target_endpoints": {},
    }

    # Collect target endpoints from session state
    endpoint_ids = st.session_state.get("arena_endpoints", ["ep_1", "ep_2"])
    for ep_id in endpoint_ids:
        name = st.session_state.get(f"arena_ep_name_{ep_id}", "")
        if not name:
            continue

        provider = st.session_state.get(f"arena_ep_provider_{ep_id}", "DashScope")
        base_url = st.session_state.get(f"arena_ep_url_{ep_id}", "") or DEFAULT_API_ENDPOINTS.get(provider, "")

        model_option = st.session_state.get(f"arena_ep_model_{ep_id}", "")
        model = (
            st.session_state.get(f"arena_ep_custom_model_{ep_id}", "") if model_option == "Custom..." else model_option
        )

        panel_config["target_endpoints"][name] = {
            "base_url": base_url,
            "api_key": st.session_state.get(f"arena_ep_key_{ep_id}", ""),
            "model": model,
            "system_prompt": st.session_state.get(f"arena_ep_system_{ep_id}", ""),
        }

    return sidebar_config, panel_config


def _render_task_config(config: dict[str, Any]) -> None:
    """Render task configuration section."""
    st.markdown(
        f"""<div class="section-header">
            <span style="margin-right: 0.5rem;">📋</span>{t("arena.config.task")}
        </div>""",
        unsafe_allow_html=True,
    )

    task_description = st.text_area(
        t("arena.config.task_description"),
        placeholder=t("arena.config.task_description_placeholder"),
        height=100,
        key="arena_task_description",
        help=t("arena.config.task_description_help"),
    )

    task_scenario = st.text_input(
        t("arena.config.task_scenario"),
        placeholder=t("arena.config.task_scenario_placeholder"),
        key="arena_task_scenario",
        help=t("arena.config.task_scenario_help"),
    )

    config["task_description"] = task_description
    config["task_scenario"] = task_scenario


def _render_single_endpoint(
    endpoint_id: str,
    endpoint_num: int,
    remove_callback: Any = None,  # pylint: disable=unused-argument
) -> dict[str, Any] | None:
    """Render a single endpoint configuration.

    Args:
        endpoint_id: Unique identifier for this endpoint
        endpoint_num: Display number
        remove_callback: Callback for remove button

    Returns:
        Endpoint configuration dict or None if removed
    """
    # Header row: Title + Remove button
    col_header, col_remove = st.columns([4, 1])
    with col_header:
        st.markdown(
            f'<div style="font-weight: 600; color: #94A3B8; font-size: 0.9rem;">'
            f'{t("arena.config.target_model")} {endpoint_num}</div>',
            unsafe_allow_html=True,
        )
    with col_remove:
        if endpoint_num > 1:  # Keep at least one endpoint
            if st.button("✕", key=f"remove_{endpoint_id}", help=t("arena.config.remove_target")):
                return None

    # Row 1: Provider + Model
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            t("api.provider"),
            options=list(DEFAULT_API_ENDPOINTS.keys()),
            key=f"arena_ep_provider_{endpoint_id}",
        )

    with col2:
        # Use stable value for custom option to survive UI language switch
        CUSTOM_VALUE = "_custom_"

        # Check if there's a custom model value already set (e.g., from preset)
        current_model = st.session_state.get(f"arena_ep_model_{endpoint_id}", "")

        # Build options list - include current model if it's custom
        model_options = list(DEFAULT_MODELS)
        if current_model and current_model not in model_options and current_model != CUSTOM_VALUE:
            model_options.insert(0, current_model)
        model_options.append(CUSTOM_VALUE)

        def format_model(x: str) -> str:
            return t("model.custom") if x == CUSTOM_VALUE else x

        model_option = st.selectbox(
            t("model.select"),
            options=model_options,
            format_func=format_model,
            key=f"arena_ep_model_{endpoint_id}",
        )

    # Row 1.5: Custom model input (only if "Custom..." selected)
    if model_option == CUSTOM_VALUE:
        model = st.text_input(
            t("model.custom_input"),
            placeholder=t("model.custom_placeholder"),
            key=f"arena_ep_custom_model_{endpoint_id}",
        )
    else:
        model = model_option

    # Row 2: API Key (full width)
    api_key = st.text_input(
        t("api.key"),
        type="password",
        placeholder=t("api.key_placeholder"),
        key=f"arena_ep_key_{endpoint_id}",
    )

    # Advanced settings (collapsed)
    with st.expander(f"⚙️ {t('arena.config.advanced_settings')}", expanded=False):
        # Display name (alias)
        endpoint_name = st.text_input(
            t("arena.config.display_name"),
            placeholder=model if model else f"model_{endpoint_num}",
            key=f"arena_ep_name_{endpoint_id}",
            help=t("arena.config.display_name_help"),
        )

        # Custom Endpoint URL (only for Custom provider)
        if provider == "Custom":
            st.text_input(
                t("arena.config.endpoint_url"),
                placeholder=t("api.custom_endpoint_placeholder"),
                key=f"arena_ep_url_{endpoint_id}",
            )

        # System Prompt
        system_prompt = st.text_area(
            t("arena.config.system_prompt"),
            placeholder=t("arena.config.system_prompt_placeholder"),
            height=68,
            key=f"arena_ep_system_{endpoint_id}",
        )

    # Determine base_url based on provider
    if provider == "Custom":
        base_url = st.session_state.get(f"arena_ep_url_{endpoint_id}", "")
    else:
        base_url = DEFAULT_API_ENDPOINTS[provider]

    # Use model name as default display name
    final_name = endpoint_name if endpoint_name else (model if model else f"model_{endpoint_num}")

    return {
        "name": final_name,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "system_prompt": system_prompt if system_prompt else None,
    }


def _init_endpoints_from_preset() -> None:
    """Initialize endpoint widgets from preset data if available."""
    preset_endpoints = st.session_state.get("arena_preset_endpoints")
    if not preset_endpoints:
        return

    # Clear existing endpoints and create new ones
    new_endpoint_ids = []
    for i, ep_data in enumerate(preset_endpoints):
        ep_id = f"ep_preset_{i}_{id(st.session_state)}"
        new_endpoint_ids.append(ep_id)

        # Set widget values
        st.session_state[f"arena_ep_name_{ep_id}"] = ep_data.get("name", f"model_{i+1}")

        # Determine provider from base_url
        base_url = ep_data.get("base_url", "")
        provider = "Custom"
        for prov, url in DEFAULT_API_ENDPOINTS.items():
            if url == base_url:
                provider = prov
                break

        st.session_state[f"arena_ep_provider_{ep_id}"] = provider
        if provider == "Custom":
            st.session_state[f"arena_ep_url_{ep_id}"] = base_url

        st.session_state[f"arena_ep_key_{ep_id}"] = ep_data.get("api_key", "")

        # Handle model - directly set the model name (will be added to options dynamically)
        model = ep_data.get("model", "")
        st.session_state[f"arena_ep_model_{ep_id}"] = model if model else DEFAULT_MODELS[0]

        st.session_state[f"arena_ep_system_{ep_id}"] = ep_data.get("system_prompt", "")

    st.session_state["arena_endpoints"] = new_endpoint_ids

    # Clear the preset data so it doesn't re-apply
    del st.session_state["arena_preset_endpoints"]


def _render_target_endpoints(config: dict[str, Any]) -> None:
    """Render target endpoints configuration section."""
    st.markdown(
        f"""<div class="section-header">
            <span style="margin-right: 0.5rem;">🎯</span>{t("arena.config.targets")}
        </div>""",
        unsafe_allow_html=True,
    )

    # Check if we need to initialize from preset
    if "arena_preset_endpoints" in st.session_state:
        _init_endpoints_from_preset()

    # Initialize endpoint list in session state
    if "arena_endpoints" not in st.session_state:
        st.session_state.arena_endpoints = ["ep_1", "ep_2"]  # Start with 2 endpoints

    endpoints_to_remove = []
    endpoints_config = {}

    for i, ep_id in enumerate(st.session_state.arena_endpoints):
        with st.container():
            st.markdown(
                '<div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; '
                'border-radius: 8px; margin-bottom: 0.5rem;">',
                unsafe_allow_html=True,
            )
            ep_config = _render_single_endpoint(ep_id, i + 1)
            st.markdown("</div>", unsafe_allow_html=True)

            if ep_config is None:
                endpoints_to_remove.append(ep_id)
            else:
                endpoints_config[ep_config["name"]] = ep_config

    # Remove marked endpoints
    for ep_id in endpoints_to_remove:
        st.session_state.arena_endpoints.remove(ep_id)
        st.rerun()

    # Add endpoint button
    if st.button(f"➕ {t('arena.config.add_target')}", key="arena_add_endpoint"):
        new_id = f"ep_{len(st.session_state.arena_endpoints) + 1}_{id(st.session_state)}"
        st.session_state.arena_endpoints.append(new_id)
        st.rerun()

    config["target_endpoints"] = endpoints_config


def _render_query_settings(config: dict[str, Any]) -> None:
    """Render query generation settings."""
    with st.expander(t("arena.config.query_settings"), expanded=False):
        st.markdown(
            f'<div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.5rem;">'
            f'{t("arena.config.query_settings_desc")}</div>',
            unsafe_allow_html=True,
        )

        seed_queries = st.text_area(
            t("arena.config.seed_queries"),
            placeholder=t("arena.config.seed_queries_placeholder"),
            height=80,
            key="arena_seed_queries",
        )

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                t("arena.config.temperature"),
                min_value=0.0,
                max_value=1.5,
                value=0.9,
                step=0.1,
                key="arena_query_temp",
                help=t("arena.config.temperature_help"),
            )

        with col2:
            max_similarity = st.slider(
                t("arena.config.dedup_threshold"),
                min_value=0.5,
                max_value=1.0,
                value=0.85,
                step=0.05,
                key="arena_max_similarity",
                help=t("arena.config.dedup_threshold_help"),
            )

        enable_evolution = st.checkbox(
            t("arena.config.enable_evolution"),
            value=False,
            key="arena_enable_evolution",
            help=t("arena.config.enable_evolution_help"),
        )

        config["seed_queries"] = [q.strip() for q in seed_queries.split("\n") if q.strip()] if seed_queries else []
        config["query_temperature"] = temperature
        config["max_similarity"] = max_similarity
        config["enable_evolution"] = enable_evolution


def render_config_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the main configuration panel.

    Args:
        sidebar_config: Configuration from sidebar

    Returns:
        Complete configuration dictionary
    """
    config: dict[str, Any] = {}

    # Merge sidebar config
    config.update(sidebar_config)

    # Preset management panel
    render_preset_panel(
        on_load=_on_preset_load,
        get_config=_get_current_config,
    )

    # Task configuration
    _render_task_config(config)

    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    # Target endpoints
    _render_target_endpoints(config)

    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    # Query settings
    _render_query_settings(config)

    return config
