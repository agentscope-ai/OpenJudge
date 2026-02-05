# -*- coding: utf-8 -*-
"""Translation aggregator for OpenJudge Studio.

This module combines translations from all feature modules into a single
dictionary for easy lookup.

Directory structure:
    translations/
    ├── __init__.py      # This file - aggregates all translations
    ├── common.py        # Common UI translations (navigation, shared components)
    ├── grader.py        # Grader feature translations
    ├── auto_arena.py    # Auto Arena feature translations
    ├── paper_review.py  # Paper Review feature translations
    └── auto_rubric.py   # Auto Rubric feature translations

To add translations for a new feature:
    1. Create a new file (e.g., `new_feature.py`)
    2. Define EN and ZH dictionaries with feature-prefixed keys
    3. Import and merge in this file's get_all_translations()
"""

from typing import Any

from shared.i18n.translations.auto_arena import EN as AUTO_ARENA_EN
from shared.i18n.translations.auto_arena import ZH as AUTO_ARENA_ZH
from shared.i18n.translations.auto_rubric import EN as AUTO_RUBRIC_EN
from shared.i18n.translations.auto_rubric import ZH as AUTO_RUBRIC_ZH
from shared.i18n.translations.common import EN as COMMON_EN
from shared.i18n.translations.common import ZH as COMMON_ZH
from shared.i18n.translations.grader import EN as GRADER_EN
from shared.i18n.translations.grader import ZH as GRADER_ZH
from shared.i18n.translations.paper_review import EN as PAPER_REVIEW_EN
from shared.i18n.translations.paper_review import ZH as PAPER_REVIEW_ZH


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
        "en": _merge_dicts(COMMON_EN, GRADER_EN, AUTO_ARENA_EN, PAPER_REVIEW_EN, AUTO_RUBRIC_EN),
        "zh": _merge_dicts(COMMON_ZH, GRADER_ZH, AUTO_ARENA_ZH, PAPER_REVIEW_ZH, AUTO_RUBRIC_ZH),
    }


__all__ = ["get_all_translations"]
