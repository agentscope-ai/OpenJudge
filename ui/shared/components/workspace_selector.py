# -*- coding: utf-8 -*-
"""Workspace selector component for multi-user isolation."""

import os

import streamlit as st
from shared.i18n import get_available_languages, get_ui_language, set_ui_language, t
from shared.services.workspace_manager import (
    WorkspaceManager,
    get_current_workspace,
    set_current_workspace,
)

# Session state keys for workspace UI
STATE_SHOW_CREATE_DIALOG = "workspace_show_create_dialog"
STATE_SHOW_DELETE_CONFIRM = "workspace_show_delete_confirm"

# Environment variable to enable shared workspaces (default: disabled for security)
# Set OPENJUDGE_ENABLE_SHARED_WORKSPACES=true to allow creating named workspaces
ENABLE_SHARED_WORKSPACES = os.environ.get("OPENJUDGE_ENABLE_SHARED_WORKSPACES", "").lower() in ("true", "1", "yes")


def _save_language_to_storage(lang: str) -> None:
    """Save language to browser localStorage via JavaScript."""
    js_code = f"""
    <script>
        localStorage.setItem('openjudge_ui_language', '{lang}');
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


def render_workspace_selector(show_language_selector: bool = False) -> None:
    """Render the workspace selector in the sidebar.

    Args:
        show_language_selector: If True, show language selector in the same row

    Allows users to:
    - See current workspace
    - Switch to a different workspace
    - Create new named workspaces
    - Switch language (if show_language_selector=True)
    """
    manager = WorkspaceManager.get_instance()
    current_ws = get_current_workspace()

    # Main selector row - adjust columns based on whether language selector is shown
    if show_language_selector:
        col_label, col_ws, col_menu, col_lang = st.columns([1.0, 2.0, 0.8, 1.5])
    else:
        col_label, col_ws, col_menu = st.columns([1.2, 3.5, 1.3])

    with col_label:
        st.markdown(
            f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748B; '
            f'padding-top: 0.5rem;">{t("workspace.label")}</div>',
            unsafe_allow_html=True,
        )

    with col_ws:
        # Build options based on whether shared workspaces are enabled
        options = []
        option_display = {}

        # Add anonymous option (current user's default) - always available
        anon_key = "__anonymous__"
        options.append(anon_key)
        option_display[anon_key] = f"🔒 {t('workspace.anonymous')}"

        # Only list named workspaces if shared workspaces are enabled
        ws_names = []
        if ENABLE_SHARED_WORKSPACES:
            workspaces = manager.list_workspaces(include_anonymous=False)
            ws_names = [ws["name"] for ws in workspaces]
            for ws in workspaces:
                options.append(ws["name"])
                option_display[ws["name"]] = f"📁 {ws['display_name']}"

        # Determine current index
        if current_ws.startswith(manager.ANONYMOUS_PREFIX):
            current_index = 0
        elif current_ws in ws_names:
            current_index = ws_names.index(current_ws) + 1
        else:
            current_index = 0

        # Only show selector if there are multiple options, otherwise just show label
        if len(options) > 1:
            selected = st.selectbox(
                t("workspace.select"),
                options=options,
                index=current_index,
                format_func=lambda x: option_display.get(x, x),
                key="workspace_selector",
                label_visibility="collapsed",
            )
        else:
            # Single option - just display it without dropdown
            st.markdown(
                f'<div style="padding: 0.5rem 0; color: #F1F5F9; font-size: 0.9rem;">'
                f"{option_display[anon_key]}</div>",
                unsafe_allow_html=True,
            )
            selected = anon_key

        # Handle workspace switch
        if selected == anon_key:
            browser_id = st.session_state.get("workspace_browser_id", "default")
            target_ws = manager.get_anonymous_workspace_name(browser_id)
        else:
            target_ws = selected

        if target_ws != current_ws:
            set_current_workspace(target_ws)
            manager.ensure_workspace_exists(target_ws)
            st.rerun()

    with col_menu:
        # Only show workspace management menu if shared workspaces are enabled
        if ENABLE_SHARED_WORKSPACES:
            with st.popover("⚙️", use_container_width=True):
                st.markdown(f"**{t('workspace.manage')}**")

                # Create new workspace
                if st.button(f"➕ {t('workspace.create')}", key="ws_create_btn", use_container_width=True):
                    st.session_state[STATE_SHOW_CREATE_DIALOG] = True

                # Delete workspace (only if named workspace is selected)
                is_named_ws = not current_ws.startswith(manager.ANONYMOUS_PREFIX)
                if st.button(
                    f"🗑️ {t('workspace.delete')}",
                    key="ws_delete_btn",
                    disabled=not is_named_ws,
                    use_container_width=True,
                    help=t("workspace.delete_help") if not is_named_ws else None,
                ):
                    st.session_state[STATE_SHOW_DELETE_CONFIRM] = current_ws
        else:
            # Show empty placeholder to maintain layout
            st.empty()

    # Language selector (if enabled)
    if show_language_selector:
        with col_lang:
            current_lang = get_ui_language()
            languages = get_available_languages()
            lang_options = list(languages.keys())

            # Initialize the selector key in session state to match current language
            if "_ui_lang_selector_inline" not in st.session_state:
                st.session_state["_ui_lang_selector_inline"] = current_lang

            selected_lang = st.selectbox(
                "Language",
                options=lang_options,
                format_func=lambda x: f"🌐 {languages[x]}",
                key="_ui_lang_selector_inline",
                label_visibility="collapsed",
            )

            # Handle language change
            if selected_lang != current_lang:
                set_ui_language(selected_lang)
                _save_language_to_storage(selected_lang)
                st.rerun()

    # Dialogs (only if shared workspaces are enabled)
    if ENABLE_SHARED_WORKSPACES:
        _render_create_dialog(manager)
        _render_delete_confirm(manager)


def _render_create_dialog(manager: WorkspaceManager) -> None:
    """Render the create workspace dialog."""
    if not st.session_state.get(STATE_SHOW_CREATE_DIALOG):
        return

    st.markdown("---")
    st.markdown(f"**{t('workspace.create_title')}**")

    new_name = st.text_input(
        t("workspace.name"),
        placeholder="my_workspace",
        help=t("workspace.name_help"),
        key="ws_new_name",
    )

    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        if st.button(f"✓ {t('workspace.create')}", key="ws_confirm_create", type="primary"):
            if not new_name:
                st.error(t("workspace.error.name_empty"))
                return

            is_valid, error_key = manager.validate_workspace_name(new_name)
            if not is_valid:
                st.error(t(error_key))
                return

            success, error_key = manager.create_workspace(new_name)
            if success:
                set_current_workspace(new_name)
                st.session_state[STATE_SHOW_CREATE_DIALOG] = False
                st.rerun()
            else:
                st.error(t(error_key))

    with col2:
        if st.button(t("common.cancel"), key="ws_cancel_create"):
            st.session_state[STATE_SHOW_CREATE_DIALOG] = False
            st.rerun()


def _render_delete_confirm(manager: WorkspaceManager) -> None:
    """Render the delete confirmation dialog."""
    ws_to_delete = st.session_state.get(STATE_SHOW_DELETE_CONFIRM)
    if not ws_to_delete:
        return

    st.warning(f"⚠️ {t('workspace.delete_confirm', name=ws_to_delete)}")

    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        if st.button(f"🗑️ {t('workspace.delete')}", key="ws_confirm_delete", type="primary"):
            success, error_key = manager.delete_workspace(ws_to_delete)
            if success:
                # Switch to anonymous workspace
                browser_id = st.session_state.get("workspace_browser_id", "default")
                anonymous_ws = manager.get_anonymous_workspace_name(browser_id)
                set_current_workspace(anonymous_ws)
                manager.ensure_workspace_exists(anonymous_ws)
                st.session_state[STATE_SHOW_DELETE_CONFIRM] = None
                st.rerun()
            else:
                st.error(t(error_key))

    with col2:
        if st.button(t("common.cancel"), key="ws_cancel_delete"):
            st.session_state[STATE_SHOW_DELETE_CONFIRM] = None
            st.rerun()
