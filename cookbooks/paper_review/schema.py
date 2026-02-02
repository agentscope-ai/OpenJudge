# -*- coding: utf-8 -*-
"""Data models for paper review results."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Reference verification status."""

    VERIFIED = "verified"
    SUSPECT = "suspect"
    ERROR = "error"


class ReviewStage(str, Enum):
    """Pipeline review stages."""

    NOT_STARTED = "not_started"
    LOADING_PDF = "loading_pdf"
    SAFETY_CHECK = "safety_check"
    CORRECTNESS = "correctness"
    REVIEW = "review"
    CRITICALITY = "criticality"
    BIB_VERIFICATION = "bib_verification"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewProgress(BaseModel):
    """Progress state for paper review pipeline."""

    stage: ReviewStage = ReviewStage.NOT_STARTED
    stage_name: str = ""
    stage_description: str = ""
    total_stages: int = 5
    completed_stages: int = 0
    progress_percent: float = 0.0
    error: Optional[str] = None

    def reset(self, total_stages: int = 5) -> None:
        """Reset progress state for a new review."""
        self.stage = ReviewStage.NOT_STARTED
        self.stage_name = ""
        self.stage_description = ""
        self.total_stages = total_stages
        self.completed_stages = 0
        self.progress_percent = 0.0
        self.error = None

    def update(
        self,
        stage: ReviewStage,
        stage_name: str,
        stage_description: str,
        completed_stages: int,
        total_stages: Optional[int] = None,
    ) -> None:
        """Update progress state.

        Args:
            stage: Current review stage
            stage_name: Display name for the stage
            stage_description: Description of what the stage does
            completed_stages: Number of completed stages
            total_stages: Optional total stages count (updates if provided)
        """
        if total_stages is not None:
            self.total_stages = total_stages
        self.stage = stage
        self.stage_name = stage_name
        self.stage_description = stage_description
        self.completed_stages = completed_stages
        self.progress_percent = (completed_stages / self.total_stages) * 100 if self.total_stages > 0 else 0

    def mark_completed(self) -> None:
        """Mark progress as completed.

        Note: Preserves stage_name and stage_description from the last stage,
        so UI knows what the final stage was before completion.
        """
        self.stage = ReviewStage.COMPLETED
        # Don't overwrite stage_name/stage_description - keep the last stage info
        self.completed_stages = self.total_stages
        self.progress_percent = 100.0
        self.error = None

    def mark_failed(self, error: str) -> None:
        """Mark progress as failed.

        Note: Preserves stage_name and stage_description from the current stage,
        so UI knows which stage failed.

        Args:
            error: Error message describing the failure
        """
        self.stage = ReviewStage.FAILED
        # Don't overwrite stage_name/stage_description - keep them so UI knows where it failed
        self.error = error


# Type alias for progress callback
ProgressCallback = Callable[["ReviewProgress"], None]


class CorrectnessResult(BaseModel):
    """Result of correctness detection."""

    score: int = Field(ge=1, le=3)
    reasoning: str
    key_issues: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Result of paper review."""

    score: int = Field(ge=1, le=6)
    review: str


class CriticalityIssues(BaseModel):
    """Classified issues from criticality verification."""

    major: List[str] = Field(default_factory=list)
    minor: List[str] = Field(default_factory=list)
    false_positives: List[str] = Field(default_factory=list)


class CriticalityResult(BaseModel):
    """Result of criticality verification."""

    score: int = Field(ge=1, le=3)
    reasoning: str
    issues: CriticalityIssues


class BibVerificationSummary(BaseModel):
    """Summary of .bib file verification."""

    total_references: int = 0
    verified: int = 0
    suspect: int = 0
    errors: int = 0
    verification_rate: float = 0.0
    suspect_references: List[str] = Field(default_factory=list)


class TexPackageInfo(BaseModel):
    """Information about processed TeX package."""

    main_tex: str
    total_files: int
    bib_files: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)


class PaperReviewResult(BaseModel):
    """Complete paper review result."""

    is_safe: bool = True
    safety_issues: List[str] = Field(default_factory=list)
    correctness: Optional[CorrectnessResult] = None
    review: Optional[ReviewResult] = None
    criticality: Optional[CriticalityResult] = None
    format_compliant: Optional[bool] = None
    bib_verification: Optional[Dict[str, BibVerificationSummary]] = None
    tex_info: Optional[TexPackageInfo] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
