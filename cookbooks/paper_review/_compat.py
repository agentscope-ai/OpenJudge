# -*- coding: utf-8 -*-
"""Compatibility layer: use openjudge base classes when available, otherwise
provide lightweight fallbacks so paper_review works without ``pip install -e .``

When the full openjudge package is installed, this module re-exports the real
classes.  When it is *not* installed, it defines minimal stubs that satisfy the
paper_review graders (which override ``aevaluate`` and only need ``self.name``
and ``self.model`` from the base hierarchy).
"""

try:
    from openjudge.graders.schema import (  # noqa: F401
        GraderError,
        GraderMode,
        GraderScore,
    )
    from openjudge.graders.llm_grader import LLMGrader  # noqa: F401
    from openjudge.models.base_chat_model import BaseChatModel  # noqa: F401

    HAS_OPENJUDGE = True

except ImportError:
    from enum import Enum
    from typing import Any, Dict

    from pydantic import BaseModel, Field

    HAS_OPENJUDGE = False

    class GraderMode(str, Enum):  # type: ignore[no-redef]
        POINTWISE = "pointwise"
        LISTWISE = "listwise"

    class _GraderResult(BaseModel):
        name: str = Field(description="The name of the grader")
        reason: str = Field(default="", description="The reason for the result")
        metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="The metadata of the grader result",
        )

    class GraderScore(_GraderResult):  # type: ignore[no-redef]
        reason: str = Field(description="reason")
        score: float = Field(description="score")

    class GraderError(_GraderResult):  # type: ignore[no-redef]
        error: str = Field(description="error")

    class BaseChatModel:  # type: ignore[no-redef]
        """Minimal stub used only as a type hint."""

    class LLMGrader:  # type: ignore[no-redef]
        """Lightweight stub matching the subset of LLMGrader that paper_review
        graders actually rely on (``self.name`` and ``self.model``)."""

        def __init__(
            self,
            name: str = "",
            mode: GraderMode = GraderMode.POINTWISE,
            description: str = "",
            model: Any = None,
            template: Any = "",
            **kwargs: Any,
        ):
            self.name = name
            self.mode = mode
            self.description = description
            self.model = model


__all__ = [
    "GraderError",
    "GraderMode",
    "GraderScore",
    "LLMGrader",
    "BaseChatModel",
    "HAS_OPENJUDGE",
]
