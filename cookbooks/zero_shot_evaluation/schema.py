# -*- coding: utf-8 -*-
"""Data schemas and configuration loading for zero-shot evaluation.

This module provides:
- Data models for configuration (OpenAIEndpoint, ZeroShotConfig, etc.)
- Configuration loading utilities (load_config, resolve_env_vars)
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# Data Models
# =============================================================================


class OpenAIEndpoint(BaseModel):
    """OpenAI-compatible endpoint configuration.

    This schema is used for all endpoint configurations including:
    - Target model endpoints
    - Judge model endpoint
    - Query generation endpoint (optional)
    """

    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key, supports ${ENV_VAR} format")
    model: str = Field(..., description="Model name")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    extra_params: Optional[Dict[str, Any]] = Field(default=None, description="Extra request parameters")


class TaskConfig(BaseModel):
    """Task configuration."""

    description: str = Field(..., description="Task description")
    scenario: Optional[str] = Field(default=None, description="Usage scenario")


class QueryGenerationConfig(BaseModel):
    """Query generation configuration."""

    num_queries: int = Field(default=20, description="Number of queries to generate")
    seed_queries: Optional[List[str]] = Field(default=None, description="Seed queries for generation")
    categories: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query categories")

    # Endpoint configuration (optional, defaults to judge_endpoint if not specified)
    # Uses OpenAIEndpoint for consistency
    endpoint: Optional[OpenAIEndpoint] = Field(
        default=None,
        description="Custom endpoint for query generation. If not set, uses judge_endpoint.",
    )

    # Diversity control parameters
    temperature: float = Field(default=0.9, ge=0.0, le=2.0, description="Sampling temperature for diversity")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")

    # Batch generation parameters
    queries_per_call: int = Field(default=10, ge=1, le=50, description="Number of queries to generate per API call")
    num_parallel_batches: int = Field(default=3, ge=1, description="Number of parallel batches")
    max_similarity: float = Field(default=0.85, ge=0.0, le=1.0, description="Max similarity threshold for dedup")

    # Evol-Instruct parameters
    enable_evolution: bool = Field(default=False, description="Enable complexity evolution")
    evolution_rounds: int = Field(default=1, ge=0, le=3, description="Number of evolution rounds")
    complexity_levels: List[str] = Field(
        default=["constraints", "reasoning", "edge_cases"],
        description="Complexity evolution strategies",
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    max_concurrency: int = Field(default=10, description="Maximum concurrency")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    retry_times: int = Field(default=3, description="Number of retries")


class OutputConfig(BaseModel):
    """Output configuration."""

    save_queries: bool = Field(default=True, description="Save generated queries")
    save_responses: bool = Field(default=True, description="Save all responses")
    save_details: bool = Field(default=True, description="Save detailed results")
    output_dir: str = Field(default="./evaluation_results", description="Output directory")


class ChartConfig(BaseModel):
    """Chart generation configuration."""

    enabled: bool = Field(default=True, description="Whether to generate win rate chart")
    title: Optional[str] = Field(default=None, description="Chart title (auto-generated if not set)")
    figsize: tuple = Field(default=(12, 7), description="Figure size (width, height) in inches")
    dpi: int = Field(default=150, ge=72, le=300, description="Image resolution")
    format: Literal["png", "svg", "pdf"] = Field(default="png", description="Output format")
    show_values: bool = Field(default=True, description="Show values on top of bars")
    highlight_best: bool = Field(default=True, description="Highlight the best model with accent color")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    enabled: bool = Field(default=False, description="Whether to generate report")
    language: Literal["zh", "en"] = Field(default="zh", description="Report language: zh | en")
    include_examples: int = Field(default=3, ge=1, le=10, description="Examples per section")
    chart: ChartConfig = Field(default_factory=ChartConfig, description="Chart configuration")


class ZeroShotConfig(BaseModel):
    """Complete zero-shot evaluation configuration."""

    task: TaskConfig
    target_endpoints: Dict[str, OpenAIEndpoint]
    judge_endpoint: OpenAIEndpoint
    query_generation: QueryGenerationConfig = Field(default_factory=QueryGenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


class GeneratedQuery(BaseModel):
    """Generated query item."""

    query: str = Field(..., description="The query text")
    category: Optional[str] = Field(default=None, description="Query category")
    difficulty: Optional[str] = Field(default=None, description="Query difficulty")


class QueryGenerationOutput(BaseModel):
    """Output schema for query generation."""

    queries: List[GeneratedQuery] = Field(..., description="List of generated queries")
    reason: str = Field(default="", description="Generation reasoning")


class ComparisonDetail(BaseModel):
    """Single pairwise comparison detail."""

    query: str = Field(..., description="Original query")
    model_a: str = Field(..., description="Model A name")
    model_b: str = Field(..., description="Model B name")
    response_a: str = Field(..., description="Model A response")
    response_b: str = Field(..., description="Model B response")
    winner: str = Field(..., description="Winner: model_a | model_b")
    score: float = Field(..., description="Score (1.0=A wins, 0.0=B wins)")
    reason: str = Field(default="", description="Evaluation reason")
    order: str = Field(default="original", description="Comparison order: original | swapped")


# =============================================================================
# Configuration Loading
# =============================================================================


def resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports ${VAR_NAME} format.

    Args:
        value: Configuration value (can be str, dict, or list)

    Returns:
        Value with environment variables resolved
    """
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


def load_config(config_path: Union[str, Path]) -> ZeroShotConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated ZeroShotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Resolve environment variables
    resolved_config = resolve_env_vars(raw_config)

    # Validate and create config object
    config = ZeroShotConfig(**resolved_config)
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Task: {config.task.description}")
    logger.info(f"Target endpoints: {list(config.target_endpoints.keys())}")

    return config


def config_to_dict(config: ZeroShotConfig) -> Dict[str, Any]:
    """Convert ZeroShotConfig to dictionary (for serialization).

    Args:
        config: ZeroShotConfig object

    Returns:
        Dictionary representation
    """
    return config.model_dump()
