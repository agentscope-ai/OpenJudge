# -*- coding: utf-8 -*-
"""Preset management panel for Auto Arena.

Provides UI for managing configuration presets:
- Load/Save presets
- Import/Export YAML files
- Delete presets
"""

from typing import Any, Callable

import streamlit as st
from features.auto_arena.services.preset_manager import PresetManager

# Session state keys
STATE_LOADED_PRESET = "arena_loaded_preset"  # Currently loaded preset name
STATE_PENDING_DELETE = "arena_pending_delete"  # Preset waiting for delete confirmation
STATE_SHOW_SAVE_DIALOG = "arena_show_save_dialog"


def _get_manager() -> PresetManager:
    """Get or create the preset manager instance."""
    if "arena_preset_manager" not in st.session_state:
        st.session_state["arena_preset_manager"] = PresetManager()
    return st.session_state["arena_preset_manager"]


def _is_builtin(manager: PresetManager, preset_name: str | None) -> bool:
    """Check if a preset is built-in."""
    if not preset_name:
        return False
    data = manager.load_preset(preset_name)
    return data.get("_ui_preset", {}).get("builtin", False) if data else False


def _render_preset_selector(
    manager: PresetManager,
    on_load: Callable[[dict[str, Any]], None],
) -> str | None:
    """Render preset dropdown. Auto-loads on selection change."""
    presets = manager.list_presets()

    if not presets:
        st.info("No presets yet")
        return None

    # Build options with display names
    options = ["--"] + [p["display_name"] for p in presets]
    name_map = {p["display_name"]: p["name"] for p in presets}

    # Find current loaded preset index
    loaded = st.session_state.get(STATE_LOADED_PRESET)
    current_index = 0
    if loaded:
        for i, p in enumerate(presets):
            if p["name"] == loaded:
                current_index = i + 1
                break

    selected_display = st.selectbox(
        "Preset",
        options=options,
        index=current_index,
        key="arena_preset_dropdown",
        label_visibility="collapsed",
    )

    selected_name = name_map.get(selected_display) if selected_display != "--" else None

    # Auto-load when selection changes
    if selected_name and selected_name != loaded:
        preset_data = manager.load_preset(selected_name)
        if preset_data:
            ui_state = PresetManager.config_to_ui_state(preset_data)
            on_load(ui_state)
            st.session_state[STATE_LOADED_PRESET] = selected_name
            st.rerun()

    return selected_name


def _render_save_dialog(get_config: Callable[[], tuple[dict, dict]]) -> None:
    """Render save dialog for entering preset name."""
    if not st.session_state.get(STATE_SHOW_SAVE_DIALOG):
        return

    st.markdown("---")
    st.markdown("**Save Preset**")

    new_name = st.text_input(
        "Name",
        placeholder="my_preset",
        help="Letters, numbers, underscores, hyphens only",
        key="arena_save_name",
    )

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Save", key="confirm_save", type="primary"):
            if not new_name:
                st.error("Name required")
                return

            is_valid, error_msg = PresetManager.validate_name(new_name)
            if not is_valid:
                st.error(error_msg)
                return

            sidebar_cfg, panel_cfg = get_config()
            config = PresetManager.ui_state_to_config(sidebar_cfg, panel_cfg)

            manager = _get_manager()
            # Allow overwrite if saving with same name
            success, error = manager.save_preset(new_name, config, overwrite=True)

            if success:
                st.session_state[STATE_LOADED_PRESET] = new_name
                st.session_state[STATE_SHOW_SAVE_DIALOG] = False
                st.rerun()
            else:
                st.error(error)

    with col2:
        if st.button("Cancel", key="cancel_save"):
            st.session_state[STATE_SHOW_SAVE_DIALOG] = False
            st.rerun()


def _render_delete_confirm() -> None:
    """Render delete confirmation dialog."""
    pending = st.session_state.get(STATE_PENDING_DELETE)
    if not pending:
        return

    st.warning(f"⚠️ Delete **{pending}**? This cannot be undone.")

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Delete", key="confirm_delete", type="primary"):
            manager = _get_manager()
            success, error = manager.delete_preset(pending)

            if success:
                # Clear loaded preset if it was the deleted one
                if st.session_state.get(STATE_LOADED_PRESET) == pending:
                    st.session_state[STATE_LOADED_PRESET] = None
                st.session_state[STATE_PENDING_DELETE] = None
                st.rerun()
            else:
                st.error(error)

    with col2:
        if st.button("Cancel", key="cancel_delete"):
            st.session_state[STATE_PENDING_DELETE] = None
            st.rerun()


def render_preset_panel(
    on_load: Callable[[dict[str, Any]], None],
    get_config: Callable[[], tuple[dict, dict]],
) -> None:
    """Render the preset management panel."""
    manager = _get_manager()

    # Main row: Label + Selector + Menu
    col_label, col_select, col_menu = st.columns([1.2, 4, 0.8])

    with col_label:
        st.markdown(
            '<div style="font-size: 0.85rem; font-weight: 600; color: #94A3B8; '
            'padding-top: 0.5rem;">📁 Presets</div>',
            unsafe_allow_html=True,
        )

    with col_select:
        selected = _render_preset_selector(manager, on_load)

    with col_menu:
        with st.popover("⚙️", use_container_width=True):
            loaded = st.session_state.get(STATE_LOADED_PRESET)
            loaded_is_builtin = _is_builtin(manager, loaded)
            selected_is_builtin = _is_builtin(manager, selected)

            # Unified Save button
            # - No preset loaded or builtin loaded → show dialog for name
            # - Non-builtin loaded → overwrite directly
            st.markdown("**Save**")
            if st.button("💾 Save", key="btn_save", use_container_width=True):
                if loaded and not loaded_is_builtin:
                    # Overwrite existing preset
                    sidebar_cfg, panel_cfg = get_config()
                    config = PresetManager.ui_state_to_config(sidebar_cfg, panel_cfg)
                    success, error = manager.save_preset(loaded, config, overwrite=True)
                    if success:
                        st.rerun()
                    else:
                        st.error(error)
                else:
                    # Show dialog to enter name
                    st.session_state[STATE_SHOW_SAVE_DIALOG] = True
                    st.rerun()

            # Delete (selected preset)
            if st.button(
                "🗑️ Delete",
                key="btn_delete",
                disabled=not selected or selected_is_builtin,
                use_container_width=True,
                help="Delete selected preset",
            ):
                st.session_state[STATE_PENDING_DELETE] = selected
                st.rerun()

            st.markdown("---")
            st.markdown("**Import / Export**")

            # Import - use a flag to prevent re-processing on rerun
            uploaded = st.file_uploader(
                "Import", type=["yaml", "yml"], key="arena_import", label_visibility="collapsed"
            )
            # Track which file we've already processed
            last_processed = st.session_state.get("arena_import_processed_file")
            current_file_id = uploaded.file_id if uploaded else None

            if uploaded and current_file_id != last_processed:
                success, error, config = manager.import_from_file(uploaded.read())
                if success and config:
                    ui_state = PresetManager.config_to_ui_state(config)
                    on_load(ui_state)
                    st.session_state[STATE_LOADED_PRESET] = None
                    # Mark this file as processed before rerun
                    st.session_state["arena_import_processed_file"] = current_file_id
                    st.rerun()
                else:
                    st.error(f"Import failed: {error}")
                    # Also mark as processed to avoid repeated error messages
                    st.session_state["arena_import_processed_file"] = current_file_id

            # Export
            sidebar_cfg, panel_cfg = get_config()
            config = PresetManager.ui_state_to_config(sidebar_cfg, panel_cfg)
            yaml_content = manager.export_to_yaml(config)

            st.download_button(
                "📤 Export YAML",
                data=yaml_content,
                file_name="auto_arena_config.yaml",
                mime="text/yaml",
                key="btn_export",
                use_container_width=True,
            )

    # Dialogs
    _render_save_dialog(get_config)
    _render_delete_confirm()

    st.markdown('<div class="custom-divider" style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
