# -*- coding: utf-8 -*-
"""Base class for discipline-specific paper review configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DisciplineConfig:
    """Configuration for a specific academic discipline.

    Controls how the review and correctness prompts are generated,
    including venue references, evaluation criteria emphasis, and
    discipline-specific correctness checks.
    """

    # Unique identifier used as the discipline parameter value
    id: str

    # Human-readable display name
    name: str

    # Representative top venues (conferences/journals) for this discipline
    venues: List[str] = field(default_factory=list)

    # Core evaluation dimensions with optional emphasis notes
    # Each entry: "DimensionName: description"
    evaluation_dimensions: List[str] = field(default_factory=list)

    # Discipline-specific correctness error categories to check
    correctness_categories: List[str] = field(default_factory=list)

    # Extra scoring guidance appended to the base scoring rubric
    scoring_notes: Optional[str] = None

    # Extra reviewer identity context (e.g., "You specialize in clinical trials.")
    reviewer_context: Optional[str] = None

    # Extra correctness evaluator context
    correctness_context: Optional[str] = None

    def format_venues(self) -> str:
        if not self.venues:
            return ""
        return ", ".join(self.venues)

    def format_evaluation_dimensions(self) -> str:
        return "\n\n".join(self.evaluation_dimensions)

    def format_correctness_categories(self) -> str:
        lines = []
        for i, cat in enumerate(self.correctness_categories, 1):
            lines.append(f"{i}. {cat}")
        return "\n".join(lines)
