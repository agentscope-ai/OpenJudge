# -*- coding: utf-8 -*-
"""Translation aggregator for OpenJudge Studio.

This module combines translations from all feature modules into a single
dictionary for easy lookup.

Directory structure:
    translations/
    ├── __init__.py      # This file - aggregates all translations
    ├── common.py        # Common UI translations (navigation, shared components)
    ├── grader.py        # Grader feature translations
    └── zero_shot.py     # Zero-Shot feature translations

To add translations for a new feature:
    1. Create a new file (e.g., `new_feature.py`)
    2. Define EN and ZH dictionaries with feature-prefixed keys
    3. Import and merge in this file's get_all_translations()
"""

from typing import Any

from shared.i18n.translations.common import EN as COMMON_EN
from shared.i18n.translations.common import ZH as COMMON_ZH
from shared.i18n.translations.grader import EN as GRADER_EN
from shared.i18n.translations.grader import ZH as GRADER_ZH
from shared.i18n.translations.zero_shot import EN as ZERO_SHOT_EN
from shared.i18n.translations.zero_shot import ZH as ZERO_SHOT_ZH


def _merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries into one.

    Later dictionaries override earlier ones for duplicate keys.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result: dict[str, Any] = {}
    for d in dicts:
        result.update(d)
    return result


def get_all_translations() -> dict[str, dict[str, str]]:
    """Get all translations merged by language.

    Returns:
        Dictionary with language codes as keys and translation dicts as values.
        Example: {"en": {...}, "zh": {...}}
    """
    return {
        "en": _merge_dicts(COMMON_EN, GRADER_EN, ZERO_SHOT_EN),
        "zh": _merge_dicts(COMMON_ZH, GRADER_ZH, ZERO_SHOT_ZH),
    }


__all__ = ["get_all_translations"]
