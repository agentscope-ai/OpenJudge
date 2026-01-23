# -*- coding: utf-8 -*-
"""Core i18n functionality for OpenJudge Studio.

Provides translation lookup, language switching, and UI components.
"""

from typing import Any

import streamlit as st
from shared.i18n.translations import get_all_translations

# Session state key for UI language
UI_LANGUAGE_KEY = "_ui_language"

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


def get_ui_language() -> str:
    """Get current UI language from session state.

    Returns:
        Current language code (e.g., 'zh', 'en')
    """
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


def render_language_selector(position: str = "sidebar") -> None:
    """Render language selector widget.

    Args:
        position: Where to render ('sidebar' or 'main')
    """
    current = get_ui_language()
    options = list(SUPPORTED_LANGUAGES.keys())

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
                index=options.index(current) if current in options else 0,
                key="_ui_lang_selector",
                label_visibility="collapsed",
            )
    else:
        selected = st.selectbox(
            "🌐 Language / 语言",
            options=options,
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=options.index(current) if current in options else 0,
            key="_ui_lang_selector",
        )

    if selected != current:
        set_ui_language(selected)
        st.rerun()
