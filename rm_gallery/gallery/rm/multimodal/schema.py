"""
Schema definitions for multimodal reward models
"""

from typing import List

from pydantic import BaseModel, Field


class EvaluationSteps(BaseModel):
    """
    Evaluation steps for G-Eval

    Attributes:
        steps: List of evaluation steps
    """

    steps: List[str] = Field(..., description="List of evaluation steps")


class Rubric(BaseModel):
    """
    Rubric entry for detailed scoring standards

    Attributes:
        score: Score value
        description: Description of what this score means

    Example:
        >>> rubric = Rubric(
        ...     score=10,
        ...     description="Perfect alignment with criteria"
        ... )
    """

    score: int = Field(..., description="Score value")
    description: str = Field(..., description="Description of what this score means")
