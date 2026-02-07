# -*- coding: utf-8 -*-
"""Constants and configuration values for Grader feature."""

from typing import Any

# Import global constants from shared module
from shared.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_API_ENDPOINTS,
    DEFAULT_MODELS,
    VISION_MODELS,
)

# ============================================================================
# Grader Categories
# ============================================================================

GRADER_CATEGORIES: dict[str, dict[str, Any]] = {
    "common": {
        "name": "Common",
        "name_zh": "通用评测",
        "icon": "",
        "description": "General-purpose evaluation graders",
    },
    "text": {
        "name": "Text",
        "name_zh": "文本评测",
        "icon": "",
        "description": "Text similarity and matching graders",
    },
    "format": {
        "name": "Format",
        "name_zh": "格式评测",
        "icon": "",
        "description": "Format and structure validation graders",
    },
    "code": {
        "name": "Code",
        "name_zh": "代码评测",
        "icon": "",
        "description": "Code quality and style graders",
    },
    "math": {
        "name": "Math",
        "name_zh": "数学评测",
        "icon": "",
        "description": "Mathematical expression verification",
    },
    "multimodal": {
        "name": "Multimodal",
        "name_zh": "多模态评测",
        "icon": "",
        "description": "Image and vision-language graders",
    },
    "agent": {
        "name": "Agent",
        "name_zh": "智能体评测",
        "icon": "",
        "description": "Agent tool use and trajectory graders",
    },
}

# ============================================================================
# Example Data for Different Grader Types
# ============================================================================

EXAMPLE_DATA: dict[str, dict[str, Any]] = {
    "default": {
        "query": "What is the capital of France?",
        "response": (
            "The capital of France is Paris. It is the largest city in France "
            "and serves as the country's major cultural, economic, and political center."
        ),
        "reference_response": "Paris is the capital of France.",
        "context": "",
    },
    "text_similarity": {
        "query": "",
        "response": "The quick brown fox jumps over the lazy dog.",
        "reference_response": "A fast brown fox leaps over a sleepy dog.",
        "context": "",
    },
    "code_style": {
        "query": "Write a function to calculate factorial",
        "response": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        "reference_response": "",
        "context": "",
    },
    "math_verify": {
        "query": "What is 2 + 2?",
        "response": "4",
        "reference_response": "2 + 2 = 4",
        "context": "",
    },
    "multimodal": {
        "query": "Describe this image",
        "response": "The image shows a beautiful sunset over the ocean.",
        "reference_response": "",
        "context": "Context text above the image.\n[IMAGE]\nContext text below the image.",
    },
    "agent_tool": {
        "query": "What's the weather in Beijing?",
        "response": "",
        "reference_response": "",
        "context": "",
        "tool_definitions": """[
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]""",
        "tool_calls": """[
    {
        "name": "get_weather",
        "arguments": {"location": "Beijing"}
    }
]""",
    },
}

# Re-export for backward compatibility
__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "DEFAULT_API_ENDPOINTS",
    "DEFAULT_MODELS",
    "EXAMPLE_DATA",
    "GRADER_CATEGORIES",
    "VISION_MODELS",
]
