# -*- coding: utf-8 -*-
"""Configuration for Grader feature."""

from features.grader.config.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_API_ENDPOINTS,
    DEFAULT_MODELS,
    EXAMPLE_DATA,
    GRADER_CATEGORIES,
    VISION_MODELS,
)
from features.grader.config.grader_registry import (
    GRADER_REGISTRY,
    get_all_grader_names,
    get_grader_config,
    get_graders_by_category,
)

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "DEFAULT_API_ENDPOINTS",
    "DEFAULT_MODELS",
    "EXAMPLE_DATA",
    "GRADER_CATEGORIES",
    "GRADER_REGISTRY",
    "VISION_MODELS",
    "get_all_grader_names",
    "get_grader_config",
    "get_graders_by_category",
]
