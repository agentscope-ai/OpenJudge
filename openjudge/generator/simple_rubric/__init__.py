# -*- coding: utf-8 -*-
"""Simple rubric generator module for automatic evaluation criteria generation.

This module provides a simple, task-description-based approach to generating
evaluation rubrics. It generates rubrics from task descriptions and sample
queries, without requiring labeled training data.

This is in contrast to the iterative_rubric module which learns rubrics from
preference data through an iterative refinement process.

Classes:
    SimpleRubricsGenerator: Main generator class that creates LLMGrader instances
    SimpleRubricsGeneratorConfig: Configuration for the generator
    TaskBasedRubricGenerator: Core rubric generation logic

Constants:
    DEFAULT_RUBRICS: Default fallback rubrics if generation fails
"""

from openjudge.generator.simple_rubric.generator import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
)
from openjudge.generator.simple_rubric.rubric_generator import (
    DEFAULT_RUBRICS,
    TaskBasedRubricGenerator,
)

__all__ = [
    # Main generator (creates LLMGrader)
    "SimpleRubricsGenerator",
    "SimpleRubricsGeneratorConfig",
    # Core rubric generation logic
    "TaskBasedRubricGenerator",
    "DEFAULT_RUBRICS",
]
