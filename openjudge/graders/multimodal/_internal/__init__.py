# -*- coding: utf-8 -*-
"""
Internal utilities and supporting modules for multimodal graders

This module contains all non-grader supporting code including:
- Data schemas (Pydantic models)
- Helper functions for image and text processing
- Context extraction utilities
- Custom Criteria specific utilities
"""

# Context utilities
from openjudge.graders.multimodal._internal.context_utils import (
    get_image_context,
    get_image_indices,
)

# Custom Criteria utilities
from openjudge.graders.multimodal._internal.criteria_utils import (
    construct_params_string,
    format_rubrics,
    get_score_range,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)

# Prompt rendering utilities
from openjudge.graders.multimodal._internal.prompt_utils import build_image_messages

# Schema definitions
from openjudge.graders.multimodal._internal.schema import EvaluationSteps, Rubric

# Helper functions and data types
from openjudge.models.schema.qwen.mllmImage import MLLMImage, format_image_content

__all__ = [
    # Schema
    "EvaluationSteps",
    "Rubric",
    # Data types
    "MLLMImage",
    # Helper functions
    "format_image_content",
    # Prompt rendering
    "build_image_messages",
    # Context functions
    "get_image_indices",
    "get_image_context",
    # Custom Criteria utilities
    "validate_criteria_and_evaluation_steps",
    "validate_and_sort_rubrics",
    "format_rubrics",
    "construct_params_string",
    "get_score_range",
]
