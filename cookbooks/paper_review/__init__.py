# -*- coding: utf-8 -*-
"""Academic Paper Review Cookbook for OpenJudge."""

from cookbooks.paper_review.disciplines import (
    BIOLOGY,
    CHEMISTRY,
    CS,
    DISCIPLINE_REGISTRY,
    ECONOMICS,
    ENVIRONMENTAL_SCIENCE,
    MATHEMATICS,
    MEDICINE,
    PHYSICS,
    PSYCHOLOGY,
    SOCIAL_SCIENCES,
    DisciplineConfig,
    get_discipline,
)
from cookbooks.paper_review.graders import (
    CorrectnessGrader,
    CriticalityGrader,
    FormatGrader,
    JailbreakingGrader,
    RebuttalAssessmentGrader,
    RebuttalGenerationGrader,
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
    ProgressCallback,
    RebuttalAssessmentResult,
    RebuttalConcern,
    RebuttalPointAssessment,
    RebuttalResult,
    ReviewProgress,
    ReviewResult,
    ReviewStage,
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
    "RebuttalGenerationGrader",
    "RebuttalAssessmentGrader",
    # Processors
    "BibChecker",
    "TexPackageProcessor",
    # Schema
    "PaperReviewResult",
    "CorrectnessResult",
    "ReviewResult",
    "CriticalityResult",
    "BibVerificationSummary",
    "RebuttalResult",
    "RebuttalConcern",
    "RebuttalAssessmentResult",
    "RebuttalPointAssessment",
    # Progress
    "ReviewStage",
    "ReviewProgress",
    "ProgressCallback",
    # Disciplines
    "DisciplineConfig",
    "DISCIPLINE_REGISTRY",
    "get_discipline",
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
