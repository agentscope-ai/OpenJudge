# -*- coding: utf-8 -*-
"""Academic Paper Review Cookbook for OpenJudge."""

from cookbooks.paper_review.graders import (
    CorrectnessGrader,
    CriticalityGrader,
    FormatGrader,
    JailbreakingGrader,
    ReviewGrader,
)
from cookbooks.paper_review.pipeline import PaperReviewPipeline, PipelineConfig
from cookbooks.paper_review.processors import BibChecker, TexPackageProcessor
from cookbooks.paper_review.report import generate_report
from cookbooks.paper_review.schema import (
    BibVerificationSummary,
    CorrectnessResult,
    CriticalityResult,
    PaperReviewResult,
    ReviewResult,
)

__all__ = [
    # Pipeline
    "PaperReviewPipeline",
    "PipelineConfig",
    # Report
    "generate_report",
    # Graders
    "CorrectnessGrader",
    "ReviewGrader",
    "CriticalityGrader",
    "FormatGrader",
    "JailbreakingGrader",
    # Processors
    "BibChecker",
    "TexPackageProcessor",
    # Schema
    "PaperReviewResult",
    "CorrectnessResult",
    "ReviewResult",
    "CriticalityResult",
    "BibVerificationSummary",
]
