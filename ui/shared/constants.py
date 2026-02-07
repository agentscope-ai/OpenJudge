# -*- coding: utf-8 -*-
"""Global constants for OpenJudge Studio.

This module contains all global constants that are shared across
multiple modules. Import from here to ensure consistency.
"""

# ============================================================================
# Application Metadata
# ============================================================================

APP_NAME = "OpenJudge Studio"
APP_VERSION = "0.3.0"

# ============================================================================
# Default API Endpoints
# ============================================================================

DEFAULT_API_ENDPOINTS: dict[str, str] = {
    "DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Custom": "",
}

# ============================================================================
# Default Models (OpenAI-compatible endpoints only)
# ============================================================================

DEFAULT_MODELS: list[str] = [
    # Qwen (DashScope) - default provider
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-max",
    "qwen-vl-max-latest",  # Vision model
    # OpenAI
    "gpt-5.2",
    "gpt-5-mini",
    "o3-mini",
    "o4-mini",
    # DeepSeek
    "deepseek-v3.2",
    "deepseek-r1",
]

# Vision-capable models for multimodal graders
VISION_MODELS: list[str] = [
    # OpenAI
    "gpt-4o",
    "gpt-5.2",
    # Qwen (DashScope)
    "qwen-vl-max-latest",
    "qwen-vl-plus-latest",
    "qwen3-vl-235b-a22b",
]
