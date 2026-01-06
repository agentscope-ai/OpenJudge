# -*- coding: utf-8 -*-
"""Generator module for creating graders and evaluation rubrics.

This module provides generators for automatically creating graders and
evaluation criteria based on data or task descriptions.

Submodules:
    simple_rubric: Task-description-based rubric generation (zero-shot)
    iterative_rubric: Preference-data-based rubric generation (iterative refinement)

Classes:
    BaseGraderGenerator: Abstract base class for grader generators
    GraderGeneratorConfig: Configuration for grader generation
    LLMGraderGenerator: Base class for LLM-based grader generators
    LLMGraderGeneratorConfig: Configuration for LLM grader generation

    # Simple rubric generation (from task description)
    SimpleRubricsGenerator: Main generator for simple rubric-based graders
    SimpleRubricsGeneratorConfig: Configuration for simple rubric generation
    TaskBasedRubricGenerator: Core rubric generation logic

Constants:
    DEFAULT_RUBRICS: Default fallback rubrics if generation fails
"""

from openjudge.generator.base_generator import (
    BaseGraderGenerator,
    GraderGeneratorConfig,
)
from openjudge.generator.llm_grader_generator import (
    LLMGraderGenerator,
    LLMGraderGeneratorConfig,
)

# Simple rubric generation
from openjudge.generator.simple_rubric import (
    DEFAULT_RUBRICS,
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
    TaskBasedRubricGenerator,
)

__all__ = [
    # Base classes
    "BaseGraderGenerator",
    "GraderGeneratorConfig",
    "LLMGraderGenerator",
    "LLMGraderGeneratorConfig",
    # Simple rubric generation
    "SimpleRubricsGenerator",
    "SimpleRubricsGeneratorConfig",
    "TaskBasedRubricGenerator",
    "DEFAULT_RUBRICS",
]
