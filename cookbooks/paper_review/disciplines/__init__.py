# -*- coding: utf-8 -*-
"""Academic discipline configurations for paper review.

Disciplines are ranked roughly by volume of academic publications.
Users can pass a discipline ID string or a DisciplineConfig object.
"""

from cookbooks.paper_review.disciplines.base import DisciplineConfig
from cookbooks.paper_review.disciplines.biology import BIOLOGY
from cookbooks.paper_review.disciplines.chemistry import CHEMISTRY
from cookbooks.paper_review.disciplines.cs import CS
from cookbooks.paper_review.disciplines.economics import ECONOMICS
from cookbooks.paper_review.disciplines.environmental_science import ENVIRONMENTAL_SCIENCE
from cookbooks.paper_review.disciplines.mathematics import MATHEMATICS
from cookbooks.paper_review.disciplines.medicine import MEDICINE
from cookbooks.paper_review.disciplines.physics import PHYSICS
from cookbooks.paper_review.disciplines.psychology import PSYCHOLOGY
from cookbooks.paper_review.disciplines.social_sciences import SOCIAL_SCIENCES

# Registry: discipline_id -> DisciplineConfig
# Ordered by approximate global publication volume
DISCIPLINE_REGISTRY: dict[str, DisciplineConfig] = {
    CS.id: CS,
    MEDICINE.id: MEDICINE,
    PHYSICS.id: PHYSICS,
    CHEMISTRY.id: CHEMISTRY,
    BIOLOGY.id: BIOLOGY,
    ECONOMICS.id: ECONOMICS,
    PSYCHOLOGY.id: PSYCHOLOGY,
    ENVIRONMENTAL_SCIENCE.id: ENVIRONMENTAL_SCIENCE,
    MATHEMATICS.id: MATHEMATICS,
    SOCIAL_SCIENCES.id: SOCIAL_SCIENCES,
}


def get_discipline(discipline: str | DisciplineConfig | None) -> DisciplineConfig | None:
    """Resolve a discipline from a string ID or a DisciplineConfig object.

    Args:
        discipline: A discipline ID string (e.g. "cs", "medicine"),
                    a DisciplineConfig instance, or None.

    Returns:
        The matching DisciplineConfig, or None if not found / not provided.

    Raises:
        ValueError: If a string ID is provided but not found in the registry.
    """
    if discipline is None:
        return None
    if isinstance(discipline, DisciplineConfig):
        return discipline
    result = DISCIPLINE_REGISTRY.get(discipline)
    if result is None:
        valid = ", ".join(DISCIPLINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown discipline ID '{discipline}'. Valid options are: {valid}"
        )
    return result


__all__ = [
    "DisciplineConfig",
    "DISCIPLINE_REGISTRY",
    "get_discipline",
    # Individual discipline instances
    "CS",
    "MEDICINE",
    "PHYSICS",
    "CHEMISTRY",
    "BIOLOGY",
    "ECONOMICS",
    "PSYCHOLOGY",
    "ENVIRONMENTAL_SCIENCE",
    "MATHEMATICS",
    "SOCIAL_SCIENCES",
]
