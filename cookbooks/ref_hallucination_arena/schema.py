# -*- coding: utf-8 -*-
"""Data schemas and configuration for Reference Hallucination Arena.

Provides config models, data models, and config loading utilities.
Follows the same patterns as cookbooks.auto_arena.schema.
"""

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class VerificationStatus(str, Enum):
    """Reference verification status."""

    VERIFIED = "verified"
    SUSPECT = "suspect"
    ERROR = "error"
    NOT_FOUND = "not_found"


class Discipline(str, Enum):
    """Academic disciplines for categorization."""

    COMPUTER_SCIENCE = "computer_science"
    BIOMEDICAL = "biomedical"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    SOCIAL_SCIENCE = "social_science"
    INTERDISCIPLINARY = "interdisciplinary"
    OTHER = "other"


# =============================================================================
# Endpoint & Task Config (reuse pattern from auto_arena)
# =============================================================================


class OpenAIEndpoint(BaseModel):
    """OpenAI-compatible endpoint configuration."""

    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key, supports ${ENV_VAR} format")
    model: str = Field(..., description="Model name")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    extra_params: Optional[Dict[str, Any]] = Field(default=None, description="Extra request parameters")


class TaskConfig(BaseModel):
    """Task configuration."""

    description: str = Field(..., description="Task description")
    scenario: Optional[str] = Field(default=None, description="Usage scenario")


# =============================================================================
# Dataset Config (queries come from user-provided dataset, not auto-generated)
# =============================================================================


class DatasetConfig(BaseModel):
    """Configuration for loading user-provided evaluation dataset.

    The dataset file must be JSON or JSONL format.
    See examples/queries_example.json for the expected schema.
    """

    path: str = Field(..., description="Path to JSON/JSONL dataset file")
    shuffle: bool = Field(default=False, description="Whether to shuffle queries before evaluation")
    max_queries: Optional[int] = Field(default=None, description="Max number of queries to use (None = use all)")


# =============================================================================
# Verification Config (new for this cookbook)
# =============================================================================


class VerificationConfig(BaseModel):
    """Configuration for reference verification."""

    crossref_mailto: Optional[str] = Field(default=None, description="Email for Crossref polite pool")
    pubmed_api_key: Optional[str] = Field(default=None, description="PubMed API key (optional, increases rate limit)")
    max_workers: int = Field(default=10, ge=1, le=50, description="Concurrent verification threads")
    timeout: float = Field(default=30.0, ge=5.0, description="Per-request timeout in seconds")
    verified_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Min composite score to count as VERIFIED (title+author+year)"
    )


# =============================================================================
# Evaluation, Output, Report Configs
# =============================================================================


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    max_concurrency: int = Field(default=40, description="Maximum concurrency")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    retry_times: int = Field(default=3, description="Number of retries")


class OutputConfig(BaseModel):
    """Output configuration."""

    save_queries: bool = Field(default=True)
    save_responses: bool = Field(default=True)
    save_details: bool = Field(default=True)
    output_dir: str = Field(default="./evaluation_results/ref_hallucination_arena")


class ChartConfig(BaseModel):
    """Chart generation configuration."""

    enabled: bool = Field(default=True)
    title: Optional[str] = Field(default=None)
    dpi: int = Field(default=300, ge=72, le=300)
    format: Literal["png", "svg", "pdf"] = Field(default="png")
    show_values: bool = Field(default=True)
    highlight_best: bool = Field(default=True)
    orientation: Literal["horizontal", "vertical"] = Field(default="vertical")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    enabled: bool = Field(default=True)
    language: Literal["zh", "en"] = Field(default="zh")
    include_examples: int = Field(default=3, ge=1, le=10)
    chart: ChartConfig = Field(default_factory=ChartConfig)


# =============================================================================
# Top-level Config
# =============================================================================


class RefArenaConfig(BaseModel):
    """Complete Reference Hallucination Arena configuration."""

    task: TaskConfig
    dataset: DatasetConfig
    target_endpoints: Dict[str, OpenAIEndpoint]
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


# =============================================================================
# Data Models
# =============================================================================


class Reference(BaseModel):
    """A single extracted reference."""

    key: str = Field(default="", description="BibTeX citation key")
    title: str = Field(..., description="Paper title")
    authors: Optional[str] = Field(default=None, description="Authors string")
    year: Optional[str] = Field(default=None, description="Publication year")
    journal: Optional[str] = Field(default=None, description="Journal or booktitle")
    doi: Optional[str] = Field(default=None, description="DOI identifier")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    entry_type: Optional[str] = Field(default=None, description="BibTeX entry type")


class MatchDetail(BaseModel):
    """Detailed matching information from verification."""

    title_match: float = 0.0
    author_match: float = 0.0
    year_match: bool = False
    matched_title: str = ""
    matched_authors: str = ""
    matched_year: str = ""
    matched_doi: str = ""

    # Strict per-field match flags (populated during verification)
    title_exact: bool = False  # normalized exact title match
    author_exact: bool = False  # all provided author last names present
    year_exact: bool = False  # year identical
    doi_exact: bool = False  # DOI identical


class VerificationResult(BaseModel):
    """Result of verifying a single reference."""

    reference: Reference
    status: VerificationStatus
    confidence: float = 0.0
    message: str = ""
    source: str = ""
    match_detail: Optional[MatchDetail] = None
    match_data: Optional[Dict[str, Any]] = None


class ModelVerificationResult(BaseModel):
    """Verification results for a single model on a single query."""

    model_name: str
    query: str
    category: Optional[str] = None
    discipline: Optional[str] = None
    total_refs: int = 0
    verified: int = 0
    suspect: int = 0
    errors: int = 0
    not_found: int = 0
    verification_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_confidence: float = 0.0
    completeness: float = 0.0
    results: List[VerificationResult] = Field(default_factory=list)

    # Year constraint compliance (only populated when query has year_constraint)
    has_year_constraint: bool = False
    year_constraint_desc: str = ""
    year_compliant: int = 0
    year_noncompliant: int = 0
    year_unknown: int = 0
    year_compliance_rate: float = 0.0


class DisciplineScore(BaseModel):
    """Scores for a specific discipline."""

    discipline: str
    total_refs: int = 0
    verified: int = 0
    verification_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_confidence: float = 0.0


class ModelScore(BaseModel):
    """Score for a single model based on objective verification metrics only."""

    model_name: str

    # Core metrics
    total_refs: int = 0
    verified_count: int = 0
    suspect_count: int = 0
    not_found_count: int = 0
    error_count: int = 0
    verification_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_confidence: float = 0.0
    completeness: float = 0.0

    # ---- Per-field accuracy rates (all refs as denominator) ----
    title_accuracy: float = 0.0  # % of refs whose title exactly matches a real paper
    author_accuracy: float = 0.0  # % of refs whose authors all match
    year_accuracy: float = 0.0  # % of refs whose year exactly matches
    doi_accuracy: float = 0.0  # % of refs whose DOI exactly matches
    overall_accuracy: float = 0.0  # % of refs where ALL fields match (= verification_rate)

    # Year constraint compliance (across queries that have year_constraint)
    year_constrained_refs: int = 0
    year_compliant_count: int = 0
    year_compliance_rate: float = 0.0

    # Per-discipline breakdown
    discipline_scores: Dict[str, DisciplineScore] = Field(default_factory=dict)

    # Verified source breakdown (crossref, pubmed, arxiv, dblp)
    verified_by_source: Dict[str, int] = Field(default_factory=dict)

    # Final ranking score (= overall_accuracy, used for sorting)
    overall_score: float = 0.0


class ArenaResult(BaseModel):
    """Final arena evaluation result."""

    rankings: List[Tuple[str, float]] = Field(default_factory=list)
    model_scores: Dict[str, ModelScore] = Field(default_factory=dict)
    total_queries: int = 0
    total_references: int = 0
    total_verified: int = 0
    overall_accuracy: float = 0.0


class YearConstraint(BaseModel):
    """Time constraint for reference recommendations.

    Supports three modes:
      - exact: a specific year (e.g. 2023)
      - range: a year range [min_year, max_year] (e.g. 2020-2024)
      - after/before: one-sided bound (e.g. after 2020, before 2015)

    Examples in dataset JSON:
      {"exact": 2023}
      {"min_year": 2020, "max_year": 2024}
      {"min_year": 2020}
      {"max_year": 2015}
    """

    exact: Optional[int] = Field(default=None, description="Exact year required")
    min_year: Optional[int] = Field(default=None, description="Minimum year (inclusive)")
    max_year: Optional[int] = Field(default=None, description="Maximum year (inclusive)")

    def check(self, year_str: Optional[str]) -> bool:
        """Check if a year string satisfies this constraint.

        Args:
            year_str: Year as string (e.g. "2023"). None = unknown = fail.

        Returns:
            True if year satisfies the constraint, False otherwise.
        """
        if not year_str:
            return False
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            return False

        if self.exact is not None:
            return year == self.exact
        ok = True
        if self.min_year is not None:
            ok = ok and year >= self.min_year
        if self.max_year is not None:
            ok = ok and year <= self.max_year
        return ok

    def describe(self) -> str:
        """Human-readable description of the constraint."""
        if self.exact is not None:
            return f"={self.exact}"
        parts = []
        if self.min_year is not None:
            parts.append(f">={self.min_year}")
        if self.max_year is not None:
            parts.append(f"<={self.max_year}")
        return " & ".join(parts) if parts else "none"

    @property
    def is_set(self) -> bool:
        """Whether any constraint is actually configured."""
        return self.exact is not None or self.min_year is not None or self.max_year is not None


class QueryItem(BaseModel):
    """A single evaluation query from the user-provided dataset.

    Required fields:
        - query: The prompt text asking the model to recommend references.

    Optional fields:
        - discipline: Academic discipline for verification routing.
        - num_refs: Expected number of references to recommend. Defaults to 5.
        - language: Query language (zh / en). Defaults to "zh".
        - year_constraint: Time constraint on recommended references.
          If set, references outside the specified range count as time-noncompliant.
        - metadata: Arbitrary extra info.

    Year constraint examples:
        Exact year:    {"year_constraint": {"exact": 2023}}
        Year range:    {"year_constraint": {"min_year": 2020, "max_year": 2024}}
        After a year:  {"year_constraint": {"min_year": 2020}}
        Before a year: {"year_constraint": {"max_year": 2015}}
        No constraint: omit the field or set to null
    """

    query: str = Field(..., description="The prompt text for reference recommendation")
    discipline: Optional[str] = Field(
        default=None,
        description="Academic discipline for verification routing",
    )
    num_refs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of references the model should recommend",
    )
    language: str = Field(default="zh", description="Query language: zh or en")
    year_constraint: Optional[YearConstraint] = Field(
        default=None,
        description="Time constraint on recommended references",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary extra metadata")


# =============================================================================
# Configuration Loading
# =============================================================================


def resolve_env_vars(value: Any) -> Any:
    """Resolve ${VAR_NAME} environment variables in config values."""
    if isinstance(value, str):
        pattern = r"\$\{(\w+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name, "")
            if not env_value:
                logger.warning(f"Environment variable {var_name} not set")
            value = value.replace(f"${{{var_name}}}", env_value)
        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: Union[str, Path]) -> RefArenaConfig:
    """Load and validate configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    resolved_config = resolve_env_vars(raw_config)
    config = RefArenaConfig(**resolved_config)

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Task: {config.task.description}")
    logger.info(f"Target endpoints: {list(config.target_endpoints.keys())}")

    return config
