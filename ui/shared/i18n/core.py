# -*- coding: utf-8 -*-
"""Core i18n functionality for OpenJudge Studio.

Provides translation lookup, language switching, and UI components.
"""

from typing import Any

import streamlit as st
from shared.i18n.translations import get_all_translations

# Session state key for UI language
UI_LANGUAGE_KEY = "_ui_language"
UI_LANGUAGE_INITIALIZED = "_ui_language_initialized"

# Supported languages
SUPPORTED_LANGUAGES = {
    "zh": "中文",
    "en": "English",
}

# Default language
DEFAULT_LANGUAGE = "zh"


def get_available_languages() -> dict[str, str]:
    """Get available languages.

    Returns:
        Dictionary mapping language codes to display names
    """
    return SUPPORTED_LANGUAGES.copy()


def _init_language_from_storage() -> None:
    """Initialize language from URL query parameters."""
    # Always check query params for language changes
    params = st.query_params
    if "lang" in params:
        lang = params.get("lang")
        if lang in SUPPORTED_LANGUAGES:
            current = st.session_state.get(UI_LANGUAGE_KEY)
            if current != lang:
                st.session_state[UI_LANGUAGE_KEY] = lang


def get_ui_language() -> str:
    """Get current UI language from session state.

    Returns:
        Current language code (e.g., 'zh', 'en')
    """
    _init_language_from_storage()
    return st.session_state.get(UI_LANGUAGE_KEY, DEFAULT_LANGUAGE)


def set_ui_language(lang: str) -> None:
    """Set UI language in session state.

    Args:
        lang: Language code (e.g., 'zh', 'en')
    """
    if lang in SUPPORTED_LANGUAGES:
        st.session_state[UI_LANGUAGE_KEY] = lang


def t(key: str, **kwargs: Any) -> str:
    """Get translated text for a key.

    Looks up the translation for the given key in the current language.
    Falls back to English, then returns the key itself if not found.

    Args:
        key: Translation key (e.g., "sidebar.api_settings")
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated text, or the key if translation not found

    Example:
        >>> t("sidebar.api_settings")
        'API 设置'
        >>> t("common.items_count", count=5)
        '共 5 条'
    """
    lang = get_ui_language()
    translations = get_all_translations()

    # Try current language
    lang_translations = translations.get(lang, {})
    text = lang_translations.get(key)

    # Fallback to English
    if text is None and lang != "en":
        en_translations = translations.get("en", {})
        text = en_translations.get(key)

    # Fallback to key itself
    if text is None:
        text = key

    # Apply format arguments if provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass  # Return unformatted text if formatting fails

    return text


def _save_language_to_storage(lang: str) -> None:
    """Save language to browser localStorage via JavaScript."""
    # Use st.markdown with script tag - lighter than components.html
    js_code = f"""
    <script>
        localStorage.setItem('openjudge_ui_language', '{lang}');
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


def render_language_selector(position: str = "sidebar") -> None:
    """Render language selector widget.

    Args:
        position: Where to render ('sidebar' or 'main')
    """
    current = get_ui_language()
    options = list(SUPPORTED_LANGUAGES.keys())

    # Initialize the selector key in session state to match current language
    if "_ui_lang_selector" not in st.session_state:
        st.session_state["_ui_lang_selector"] = current

    # Custom styling for compact selector
    if position == "sidebar":
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(
                '<div style="font-size: 1.2rem; padding-top: 0.25rem;">🌐</div>',
                unsafe_allow_html=True,
            )
        with col2:
            selected = st.selectbox(
                "Language",
                options=options,
                format_func=lambda x: SUPPORTED_LANGUAGES[x],
                key="_ui_lang_selector",
                label_visibility="collapsed",
            )
    else:
        selected = st.selectbox(
            "🌐 Language / 语言",
            options=options,
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            key="_ui_lang_selector",
        )

    # Handle language change
    if selected != current:
        set_ui_language(selected)
        # Save to localStorage only when language changes
        _save_language_to_storage(selected)
        st.rerun()


def inject_language_loader() -> None:
    """Initialize language from URL query parameters.

    This is a lightweight function that doesn't inject any HTML.
    Language loading is handled by _init_language_from_storage().
    """
    # Language is already initialized via _init_language_from_storage()
    # which is called by get_ui_language()


def _render_header_language_selector() -> None:
    """Placeholder for header language selector (not implemented).

    Actual rendering is done via render_sidebar_language_selector.
    This is kept as a private function for potential future implementation.
    """


def render_sidebar_language_selector() -> None:
    """Render a compact language selector in the sidebar.

    Uses Streamlit native selectbox for reliability.
    """
    current_lang = get_ui_language()
    lang_options = list(SUPPORTED_LANGUAGES.keys())

    selected_lang = st.selectbox(
        "🌐",
        options=lang_options,
        index=lang_options.index(current_lang) if current_lang in lang_options else 0,
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        key="_sidebar_lang_selector",
        label_visibility="collapsed",
    )

    # Handle language change
    if selected_lang != current_lang:
        set_ui_language(selected_lang)
        _save_language_to_storage(selected_lang)
        st.rerun()
