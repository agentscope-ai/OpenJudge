# -*- coding: utf-8 -*-
"""Internationalization (i18n) module for OpenJudge Studio.

This module provides multi-language support for the UI.
Usage:
    from shared.i18n import t, get_ui_language, set_ui_language, render_language_selector

    # Get translated text
    text = t("sidebar.api_settings")

    # Get current language
    lang = get_ui_language()

    # Set language
    set_ui_language("zh")

    # Render language selector widget
    render_language_selector()
"""

from shared.i18n.core import (
    get_available_languages,
    get_ui_language,
    inject_language_loader,
    render_language_selector,
    render_sidebar_language_selector,
    set_ui_language,
    t,
)

__all__ = [
    "t",
    "get_ui_language",
    "set_ui_language",
    "render_language_selector",
    "get_available_languages",
    "inject_language_loader",
    "render_sidebar_language_selector",
]
