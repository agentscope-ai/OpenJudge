# -*- coding: utf-8 -*-
"""Data models for paper review results."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Reference verification status."""

    VERIFIED = "verified"
    SUSPECT = "suspect"
    ERROR = "error"


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

    major: List[Any] = Field(default_factory=list)  # Can be str or dict
    minor: List[Any] = Field(default_factory=list)
    false_positives: List[Any] = Field(default_factory=list)


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
