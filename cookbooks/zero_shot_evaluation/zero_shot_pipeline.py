# -*- coding: utf-8 -*-
"""End-to-end pipeline for zero-shot evaluation.

This module provides the ZeroShotPipeline class for end-to-end evaluation
of AI models without labeled data. It integrates with OpenJudge's core
components for grading, analysis, and rubric generation.

Pipeline Steps:
    1. Generate test queries
    2. Collect responses from target endpoints
    3. Generate evaluation rubrics
    4. Run pairwise evaluation
    5. Analyze and rank results
"""

import json
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field

from cookbooks.zero_shot_evaluation.chart_generator import WinRateChartGenerator
from cookbooks.zero_shot_evaluation.query_generator import QueryGenerator
from cookbooks.zero_shot_evaluation.response_collector import ResponseCollector
from cookbooks.zero_shot_evaluation.schema import (
    ComparisonDetail,
    GeneratedQuery,
    OpenAIEndpoint,
    ZeroShotConfig,
    load_config,
)

# OpenJudge core components
from openjudge.analyzer import PairwiseAnalysisResult, PairwiseAnalyzer
from openjudge.generator.simple_rubric import TaskBasedRubricGenerator
from openjudge.graders.llm_grader import GraderMode, LLMGrader
from openjudge.graders.schema import GraderError, GraderResult
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# =============================================================================
# Checkpoint Management (integrated from checkpoint.py)
# =============================================================================


class EvaluationStage(str, Enum):
    """Evaluation pipeline stages."""

    NOT_STARTED = "not_started"
    QUERIES_GENERATED = "queries_generated"
    RESPONSES_COLLECTED = "responses_collected"
    RUBRICS_GENERATED = "rubrics_generated"
    EVALUATION_COMPLETE = "evaluation_complete"

    @classmethod
    def get_order(cls, stage: "EvaluationStage") -> int:
        """Get numeric order of a stage for comparison."""
        order = {
            cls.NOT_STARTED: 0,
            cls.QUERIES_GENERATED: 1,
            cls.RESPONSES_COLLECTED: 2,
            cls.RUBRICS_GENERATED: 3,
            cls.EVALUATION_COMPLETE: 4,
        }
        return order.get(stage, -1)

    def __ge__(self, other: "EvaluationStage") -> bool:
        """Compare stages by pipeline order, not string value."""
        return self.get_order(self) >= self.get_order(other)

    def __gt__(self, other: "EvaluationStage") -> bool:
        """Compare stages by pipeline order, not string value."""
        return self.get_order(self) > self.get_order(other)

    def __le__(self, other: "EvaluationStage") -> bool:
        """Compare stages by pipeline order, not string value."""
        return self.get_order(self) <= self.get_order(other)

    def __lt__(self, other: "EvaluationStage") -> bool:
        """Compare stages by pipeline order, not string value."""
        return self.get_order(self) < self.get_order(other)


class _CheckpointData(BaseModel):
    """Internal checkpoint data model."""

    stage: EvaluationStage = Field(default=EvaluationStage.NOT_STARTED)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Data files
    queries_file: Optional[str] = None
    responses_file: Optional[str] = None
    rubrics_file: Optional[str] = None

    # Progress tracking
    total_queries: int = 0
    collected_responses: int = 0
    evaluated_pairs: int = 0
    total_pairs: int = 0

    # Judge model info (for detecting config changes)
    judge_model: Optional[str] = None
    judge_base_url: Optional[str] = None


class _CheckpointManager:
    """Internal checkpoint manager for evaluation pipeline resume capability."""

    CHECKPOINT_FILE = "checkpoint.json"
    QUERIES_FILE = "queries.json"
    RESPONSES_FILE = "responses.json"
    RUBRICS_FILE = "rubrics.json"
    DETAILS_FILE = "comparison_details.json"

    def __init__(self, output_dir: str):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to store checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint: Optional[_CheckpointData] = None

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.CHECKPOINT_FILE

    def load(self) -> Optional[_CheckpointData]:
        """Load existing checkpoint if available."""
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._checkpoint = _CheckpointData(**data)
            logger.info(f"Loaded checkpoint: stage={self._checkpoint.stage.value}")
            return self._checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: _CheckpointData) -> None:
        """Save checkpoint to file."""
        checkpoint.updated_at = datetime.now().isoformat()
        self._checkpoint = checkpoint

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.model_dump(), f, indent=2, ensure_ascii=False)

        logger.debug(f"Checkpoint saved: stage={checkpoint.stage.value}")

    def save_queries(self, queries: List[GeneratedQuery]) -> str:
        """Save generated queries."""
        file_path = self.output_dir / self.QUERIES_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([q.model_dump() for q in queries], f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(queries)} queries to {file_path}")
        return str(file_path)

    def load_queries(self) -> List[GeneratedQuery]:
        """Load saved queries."""
        file_path = self.output_dir / self.QUERIES_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = [GeneratedQuery(**item) for item in data]
        logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries

    def save_responses(self, responses: List[Dict[str, Any]]) -> str:
        """Save collected responses."""
        file_path = self.output_dir / self.RESPONSES_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(responses)} responses to {file_path}")
        return str(file_path)

    def load_responses(self) -> List[Dict[str, Any]]:
        """Load saved responses."""
        file_path = self.output_dir / self.RESPONSES_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            responses = json.load(f)

        logger.info(f"Loaded {len(responses)} responses from {file_path}")
        return responses

    def save_rubrics(self, rubrics: List[str]) -> str:
        """Save generated rubrics."""
        file_path = self.output_dir / self.RUBRICS_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(rubrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(rubrics)} rubrics to {file_path}")
        return str(file_path)

    def load_rubrics(self) -> List[str]:
        """Load saved rubrics."""
        file_path = self.output_dir / self.RUBRICS_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            rubrics = json.load(f)

        logger.info(f"Loaded {len(rubrics)} rubrics from {file_path}")
        return rubrics

    def save_comparison_details(self, details: List[ComparisonDetail]) -> str:
        """Save comparison details."""
        file_path = self.output_dir / self.DETAILS_FILE
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([d.model_dump() for d in details], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(details)} comparison details to {file_path}")
        return str(file_path)

    def load_comparison_details(self) -> List[ComparisonDetail]:
        """Load saved comparison details."""
        file_path = self.output_dir / self.DETAILS_FILE
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ComparisonDetail(**item) for item in data]

    def update_stage(
        self,
        stage: EvaluationStage,
        **kwargs,
    ) -> None:
        """Update checkpoint stage and save."""
        if self._checkpoint is None:
            self._checkpoint = _CheckpointData()

        self._checkpoint.stage = stage
        for key, value in kwargs.items():
            if hasattr(self._checkpoint, key):
                setattr(self._checkpoint, key, value)

        self.save(self._checkpoint)

    def clear(self) -> None:
        """Clear all checkpoint data."""
        for file_name in [
            self.CHECKPOINT_FILE,
            self.QUERIES_FILE,
            self.RESPONSES_FILE,
            self.RUBRICS_FILE,
            self.DETAILS_FILE,
        ]:
            file_path = self.output_dir / file_name
            if file_path.exists():
                file_path.unlink()

        self._checkpoint = None
        logger.info("Checkpoint cleared")

    def clear_comparison_details(self) -> None:
        """Clear only comparison details to re-run pairwise evaluation.

        Keeps queries, responses, and rubrics intact.
        Resets checkpoint stage to RUBRICS_GENERATED.
        """
        details_path = self.output_dir / self.DETAILS_FILE
        if details_path.exists():
            details_path.unlink()
            logger.info(f"Cleared {details_path}")

        if self._checkpoint:
            self._checkpoint.stage = EvaluationStage.RUBRICS_GENERATED
            self._checkpoint.evaluated_pairs = 0
            self._checkpoint.total_pairs = 0
            self.save(self._checkpoint)

    def is_judge_changed(self, current_model: str, current_base_url: str) -> bool:
        """Check if judge model configuration has changed.

        Args:
            current_model: Current judge model name from config
            current_base_url: Current judge base URL from config

        Returns:
            True if judge model has changed, False otherwise
        """
        if self._checkpoint is None:
            return False

        # If no previous judge info stored, assume no change (backward compatibility)
        if self._checkpoint.judge_model is None:
            return False

        return self._checkpoint.judge_model != current_model or self._checkpoint.judge_base_url != current_base_url

    def update_judge_info(self, model: str, base_url: str) -> None:
        """Update stored judge model information."""
        if self._checkpoint is None:
            self._checkpoint = _CheckpointData()
        self._checkpoint.judge_model = model
        self._checkpoint.judge_base_url = base_url
        self.save(self._checkpoint)


# =============================================================================
# Evaluation Result
# =============================================================================


class EvaluationResult(BaseModel):
    """Result of zero-shot evaluation.

    Attributes:
        rankings: List of (model_name, win_rate) tuples sorted by win rate
        win_rates: Win rate for each model
        win_matrix: Win rate matrix where win_matrix[A][B] = how often A beats B
        best_pipeline: Name of the best performing pipeline
        total_queries: Total number of queries evaluated
        total_comparisons: Total number of pairwise comparisons
    """

    rankings: List[Tuple[str, float]] = Field(default_factory=list)
    win_rates: Dict[str, float] = Field(default_factory=dict)
    win_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    best_pipeline: str = Field(default="")
    total_queries: int = Field(default=0)
    total_comparisons: int = Field(default=0)

    @classmethod
    def from_analysis(cls, analysis: PairwiseAnalysisResult, total_queries: int) -> "EvaluationResult":
        """Create EvaluationResult from PairwiseAnalysisResult.

        Args:
            analysis: Analysis result from PairwiseAnalyzer
            total_queries: Total number of queries evaluated

        Returns:
            EvaluationResult instance
        """
        return cls(
            rankings=analysis.rankings,
            win_rates=analysis.win_rates,
            win_matrix=analysis.win_matrix,
            best_pipeline=analysis.best_model,
            total_queries=total_queries,
            total_comparisons=analysis.total_comparisons,
        )


# =============================================================================
# Zero-Shot Pipeline
# =============================================================================


class ZeroShotPipeline:
    """End-to-end zero-shot evaluation pipeline with checkpoint support.

    This pipeline automates the complete evaluation process:
    1. Generate diverse test queries based on task description
    2. Collect responses from multiple target endpoints
    3. Generate evaluation rubrics using LLM
    4. Run pairwise comparisons between model responses
    5. Analyze results and rank models

    The pipeline integrates with OpenJudge's core components:
    - Uses TaskBasedRubricGenerator from openjudge.generator.simple_rubric for rubric generation
    - Uses PairwiseAnalyzer from openjudge.analyzer for result analysis
    - Uses LLMGrader and GradingRunner for pairwise evaluation

    Attributes:
        config: Pipeline configuration
        _queries: Generated queries
        _responses: Collected responses
        _rubrics: Generated rubrics

    Example:
        >>> from cookbooks.zero_shot_evaluation import ZeroShotPipeline
        >>> pipeline = ZeroShotPipeline.from_config("config.yaml")
        >>> result = await pipeline.evaluate()
        >>> print(f"Best model: {result.best_pipeline}")
    """

    def __init__(
        self,
        config: Optional[ZeroShotConfig] = None,
        *,
        task_description: Optional[str] = None,
        target_endpoints: Optional[Dict[str, OpenAIEndpoint]] = None,
        judge_endpoint: Optional[OpenAIEndpoint] = None,
        num_queries: int = 20,
        resume: bool = True,
    ):
        """Initialize ZeroShotPipeline.

        Args:
            config: Complete configuration object
            task_description: Task description (alternative to config)
            target_endpoints: Target endpoints (alternative to config)
            judge_endpoint: Judge endpoint (alternative to config)
            num_queries: Number of queries to generate
            resume: Whether to resume from checkpoint if available
        """
        if config:
            self.config = config
        else:
            if not all([task_description, target_endpoints, judge_endpoint]):
                raise ValueError("Must provide either config or all individual parameters")
            from cookbooks.zero_shot_evaluation.schema import (
                EvaluationConfig,
                QueryGenerationConfig,
                TaskConfig,
            )

            self.config = ZeroShotConfig(
                task=TaskConfig(description=task_description),
                target_endpoints=target_endpoints,
                judge_endpoint=judge_endpoint,
                query_generation=QueryGenerationConfig(num_queries=num_queries),
                evaluation=EvaluationConfig(),
            )

        self._queries: List[GeneratedQuery] = []
        self._responses: List[Dict[str, Any]] = []
        self._rubrics: List[str] = []
        self._comparison_details: List[ComparisonDetail] = []

        # Initialize checkpoint manager
        self._checkpoint_mgr = _CheckpointManager(self.config.output.output_dir)
        self._resume = resume

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "ZeroShotPipeline":
        """Create pipeline from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ZeroShotPipeline instance
        """
        config = load_config(config_path)
        return cls(config=config)

    def _create_judge_model(self) -> OpenAIChatModel:
        """Create judge model from endpoint configuration."""
        endpoint = self.config.judge_endpoint
        extra_params = endpoint.extra_params or {}
        return OpenAIChatModel(
            model=endpoint.model,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            **extra_params,
        )

    async def generate_queries(self) -> List[GeneratedQuery]:
        """Step 1: Generate test queries."""
        logger.info("Step 1: Generating test queries...")
        generator = QueryGenerator(
            judge_endpoint=self.config.judge_endpoint,
            task_config=self.config.task,
            query_config=self.config.query_generation,
        )
        self._queries = await generator.generate()
        return self._queries

    async def collect_responses(
        self,
        queries: Optional[List[GeneratedQuery]] = None,
        existing_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2: Collect responses from all target endpoints.

        Supports incremental collection: if existing_responses is provided,
        only collects from endpoints that are missing in the existing data.

        Args:
            queries: List of queries to collect responses for
            existing_responses: Optional existing responses to merge with

        Returns:
            List of response data with all endpoint responses
        """
        queries = queries or self._queries
        if not queries:
            raise ValueError("No queries available. Run generate_queries() first.")

        # Determine which endpoints need collection
        endpoints_to_collect = dict(self.config.target_endpoints)

        if existing_responses:
            # Find endpoints that already have responses
            existing_endpoints = set()
            for resp_data in existing_responses:
                existing_endpoints.update(resp_data.get("responses", {}).keys())

            # Filter to only new endpoints
            new_endpoints = {k: v for k, v in endpoints_to_collect.items() if k not in existing_endpoints}

            if new_endpoints:
                logger.info(
                    f"Incremental collection: existing={existing_endpoints}, " f"new={set(new_endpoints.keys())}"
                )
                endpoints_to_collect = new_endpoints
            else:
                logger.info("All endpoints already have responses, skipping collection")
                self._responses = existing_responses
                return self._responses

        logger.info(f"Step 2: Collecting responses from {len(endpoints_to_collect)} target endpoints...")
        collector = ResponseCollector(
            target_endpoints=endpoints_to_collect,
            evaluation_config=self.config.evaluation,
        )
        new_responses = await collector.collect(queries)

        # Merge with existing responses if provided
        if existing_responses:
            self._responses = self._merge_responses(existing_responses, new_responses)
        else:
            self._responses = new_responses

        return self._responses

    def _merge_responses(
        self,
        existing: List[Dict[str, Any]],
        new: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge existing and new responses by query.

        Args:
            existing: Existing response data
            new: New response data to merge

        Returns:
            Merged response data
        """
        # Create lookup by query
        existing_by_query = {r["query"]: r for r in existing}
        new_by_query = {r["query"]: r for r in new}

        merged = []
        all_queries = set(existing_by_query.keys()) | set(new_by_query.keys())

        for query in all_queries:
            existing_resp = existing_by_query.get(query, {})
            new_resp = new_by_query.get(query, {})

            merged_item = {
                "query": query,
                "category": existing_resp.get("category") or new_resp.get("category", ""),
                "difficulty": existing_resp.get("difficulty") or new_resp.get("difficulty", ""),
                "responses": {
                    **existing_resp.get("responses", {}),
                    **new_resp.get("responses", {}),
                },
            }
            merged.append(merged_item)

        logger.info(f"Merged responses: {len(merged)} queries with combined endpoints")
        return merged

    async def generate_rubrics(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Step 3: Generate evaluation rubrics using OpenJudge's TaskBasedRubricGenerator."""
        logger.info("Step 3: Generating evaluation rubrics...")

        if not sample_queries and self._queries:
            sample_queries = [q.query for q in self._queries[:5]]

        # Use OpenJudge's TaskBasedRubricGenerator
        generator = TaskBasedRubricGenerator(
            model=self._create_judge_model(),
            task_description=self.config.task.description,
            scenario=self.config.task.scenario,
        )
        self._rubrics = await generator.generate(sample_queries)
        return self._rubrics

    def _prepare_pairwise_data(
        self,
        responses: List[Dict[str, Any]],
    ) -> Tuple[List[dict], List[str]]:
        """Prepare pairwise comparison dataset.

        Creates comparison pairs for all model combinations, with both
        original and swapped orders to eliminate position bias.

        Supports incremental updates: if config endpoints don't match saved
        responses, uses the intersection of endpoints that have actual data.

        Args:
            responses: List of response data from collect_responses()

        Returns:
            Tuple of (dataset, endpoint_names)
        """
        config_endpoints = set(self.config.target_endpoints.keys())

        # Get endpoints that actually have responses
        response_endpoints = set()
        for resp_data in responses:
            resp_dict = resp_data.get("responses", {})
            response_endpoints.update(resp_dict.keys())

        # Use intersection if there's a mismatch, otherwise use config endpoints
        # Always sort to ensure consistent pair ordering across runs
        if config_endpoints != response_endpoints:
            available_endpoints = config_endpoints & response_endpoints
            if available_endpoints:
                logger.warning(
                    f"Endpoint mismatch detected. Config: {config_endpoints}, "
                    f"Responses: {response_endpoints}. Using intersection: {available_endpoints}"
                )
                endpoint_names = sorted(available_endpoints)
            else:
                # No overlap - use whatever endpoints exist in responses
                logger.warning(
                    f"No endpoint overlap between config and responses. "
                    f"Using response endpoints: {response_endpoints}"
                )
                endpoint_names = sorted(response_endpoints)
        else:
            endpoint_names = sorted(config_endpoints)

        pairs = list(combinations(endpoint_names, 2))

        dataset = []
        for resp_data in responses:
            query = resp_data["query"]
            resp_dict = resp_data["responses"]

            for ep_a, ep_b in pairs:
                resp_a = resp_dict.get(ep_a)
                resp_b = resp_dict.get(ep_b)

                if resp_a is None or resp_b is None:
                    continue

                # Original order
                dataset.append(
                    {
                        "evaluation_data": {
                            "instruction": query,
                            "response_a": resp_a,
                            "response_b": resp_b,
                        },
                        "metadata": {
                            "model_a": ep_a,
                            "model_b": ep_b,
                            "order": "original",
                        },
                    }
                )
                # Swapped order (to eliminate position bias)
                dataset.append(
                    {
                        "evaluation_data": {
                            "instruction": query,
                            "response_a": resp_b,
                            "response_b": resp_a,
                        },
                        "metadata": {
                            "model_a": ep_b,
                            "model_b": ep_a,
                            "order": "swapped",
                        },
                    }
                )

        return dataset, endpoint_names

    def _build_pairwise_grader(self, rubrics: List[str]) -> LLMGrader:
        """Build pairwise comparison grader."""
        rubrics_text = "\n".join(f"- {r}" for r in rubrics)

        template = PromptTemplate(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are an expert evaluator. Compare two responses based on the given criteria.\n"
                    f"Evaluation Criteria:\n{rubrics_text}\n\n"
                    "Output JSON with 'score' (1.0 if Response A is better, 0.0 if Response B is better) "
                    "and 'reason' (brief explanation).",
                ),
                ChatMessage(
                    role="user",
                    content="Query: {instruction}\n\n"
                    "Response A:\n{response_a}\n\n"
                    "Response B:\n{response_b}\n\n"
                    "Which response is better based on the criteria?",
                ),
            ],
        )

        endpoint = self.config.judge_endpoint
        extra_params = endpoint.extra_params or {}

        return LLMGrader(
            name="pairwise_comparator",
            mode=GraderMode.POINTWISE,
            model=OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                temperature=extra_params.get("temperature", 0.1),
            ),
            template=template,
        )

    async def _run_pairwise_evaluation(
        self,
        dataset: List[dict],
        rubrics: List[str],
    ) -> Tuple[List[GraderResult], List[ComparisonDetail]]:
        """Run pairwise evaluation and collect comparison details."""
        grader = self._build_pairwise_grader(rubrics)

        mapper = {
            "instruction": "evaluation_data.instruction",
            "response_a": "evaluation_data.response_a",
            "response_b": "evaluation_data.response_b",
        }

        runner = GradingRunner(
            grader_configs={
                "pairwise": GraderConfig(grader=grader, mapper=mapper),
            },
            max_concurrency=self.config.evaluation.max_concurrency,
        )

        logger.info(f"Running {len(dataset)} pairwise comparisons...")
        results = await runner.arun(dataset)
        grader_results = results["pairwise"]

        # Collect comparison details (skip GraderError results)
        details = []
        for sample, result in zip(dataset, grader_results):
            if isinstance(result, GraderError):
                continue
            score = getattr(result, "score", None)
            if score is None:
                continue
            details.append(
                ComparisonDetail(
                    query=sample["evaluation_data"]["instruction"],
                    model_a=sample["metadata"]["model_a"],
                    model_b=sample["metadata"]["model_b"],
                    response_a=sample["evaluation_data"]["response_a"],
                    response_b=sample["evaluation_data"]["response_b"],
                    winner="model_a" if score >= 0.5 else "model_b",
                    score=score,
                    reason=getattr(result, "reason", ""),
                    order=sample["metadata"].get("order", "original"),
                )
            )

        return grader_results, details

    def _analyze_results(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        endpoint_names: List[str],
    ) -> EvaluationResult:
        """Analyze pairwise comparison results using OpenJudge's PairwiseAnalyzer."""
        # Use OpenJudge's PairwiseAnalyzer for analysis
        analyzer = PairwiseAnalyzer(model_names=endpoint_names)
        analysis = analyzer.analyze(dataset, grader_results)

        # Convert to EvaluationResult
        return EvaluationResult.from_analysis(analysis, total_queries=len(self._responses))

    def _filter_existing_comparisons(
        self,
        dataset: List[dict],
        existing_details: List[ComparisonDetail],
    ) -> Tuple[List[dict], List[GraderResult]]:
        """Filter out comparisons that already exist.

        Args:
            dataset: Full dataset of comparison pairs to evaluate
            existing_details: Previously saved comparison details

        Returns:
            Tuple of (dataset_to_run, existing_grader_results)
        """
        from openjudge.graders.schema import GraderScore

        # Build lookup by normalized key for existing comparisons
        existing_by_key = {}
        for detail in existing_details:
            key = self._normalize_comparison_key(detail.query, detail.model_a, detail.model_b, detail.order)
            existing_by_key[key] = detail  # Later entries overwrite earlier

        dataset_to_run = []
        existing_results = []

        for sample in dataset:
            key = self._normalize_comparison_key(
                sample["evaluation_data"]["instruction"],
                sample["metadata"]["model_a"],
                sample["metadata"]["model_b"],
                sample["metadata"].get("order", "original"),
            )
            if key in existing_by_key:
                # Use lookup dict for O(1) access
                detail = existing_by_key[key]
                existing_results.append(
                    GraderScore(
                        name="pairwise_comparator",
                        score=detail.score,
                        reason=detail.reason,
                    )
                )
            else:
                dataset_to_run.append(sample)

        return dataset_to_run, existing_results

    def _normalize_comparison_key(
        self,
        query: str,
        model_a: str,
        model_b: str,
        order: str,
    ) -> Tuple[str, str, str, str]:
        """Normalize comparison key to ensure consistent deduplication.

        Normalizes model ordering for consistent key generation while preserving
        the order field (original/swapped) to keep both position-bias test
        comparisons distinct.
        """
        if model_a <= model_b:
            return (query, model_a, model_b, order)
        # Swap models alphabetically but keep order unchanged
        # (original, swapped) must remain distinct to test position bias
        return (query, model_b, model_a, order)

    def _deduplicate_comparison_details(
        self,
        details: List[ComparisonDetail],
    ) -> List[ComparisonDetail]:
        """Deduplicate comparison details using normalized keys.

        Handles cases where the same comparison was recorded with different
        model orderings due to inconsistent endpoint list ordering.

        Args:
            details: List of comparison details (may contain duplicates)

        Returns:
            Deduplicated list of comparison details
        """
        seen = {}
        for detail in details:
            # Use normalized key to handle ordering inconsistencies
            key = self._normalize_comparison_key(detail.query, detail.model_a, detail.model_b, detail.order)
            # Later entries overwrite earlier ones (keeps most recent)
            seen[key] = detail

        deduplicated = list(seen.values())
        if len(deduplicated) < len(details):
            logger.info(f"Deduplicated comparison details: {len(details)} -> {len(deduplicated)}")
        return deduplicated

    def _rebuild_grader_results(
        self,
        dataset: List[dict],
        all_details: List[ComparisonDetail],
    ) -> List[GraderResult]:
        """Rebuild grader results from comparison details in dataset order.

        Args:
            dataset: Full dataset (defines the expected order)
            all_details: All comparison details (existing + new)

        Returns:
            List of GraderResult in the same order as dataset
        """
        from openjudge.graders.schema import GraderScore

        # Build lookup by normalized key
        details_by_key = {}
        for detail in all_details:
            key = self._normalize_comparison_key(detail.query, detail.model_a, detail.model_b, detail.order)
            details_by_key[key] = detail

        results = []
        for sample in dataset:
            key = self._normalize_comparison_key(
                sample["evaluation_data"]["instruction"],
                sample["metadata"]["model_a"],
                sample["metadata"]["model_b"],
                sample["metadata"].get("order", "original"),
            )
            if key in details_by_key:
                detail = details_by_key[key]
                results.append(
                    GraderScore(
                        name="pairwise_comparator",
                        score=detail.score,
                        reason=detail.reason,
                    )
                )
            else:
                # This shouldn't happen, but handle gracefully
                logger.warning(f"Missing comparison detail for {key}")
                results.append(
                    GraderError(
                        name="pairwise_comparator",
                        error="Missing comparison detail",
                    )
                )

        return results

    async def evaluate(
        self,
        queries: Optional[List[GeneratedQuery]] = None,
        rubrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Run complete evaluation pipeline with checkpoint support.

        Args:
            queries: Optional pre-generated queries
            rubrics: Optional pre-generated rubrics

        Returns:
            EvaluationResult with rankings
        """
        # Try to resume from checkpoint
        checkpoint = None
        if self._resume:
            checkpoint = self._checkpoint_mgr.load()

        # Auto-detect judge model change
        current_judge_model = self.config.judge_endpoint.model
        current_judge_base_url = self.config.judge_endpoint.base_url
        if checkpoint and self._checkpoint_mgr.is_judge_changed(current_judge_model, current_judge_base_url):
            logger.info(
                f"Judge model changed (was: {checkpoint.judge_model}, "
                f"now: {current_judge_model}), re-running pairwise evaluation..."
            )
            self._checkpoint_mgr.clear_comparison_details()
            # Reload checkpoint after clearing
            checkpoint = self._checkpoint_mgr.load()

        # Step 1: Generate or load queries
        if queries:
            self._queries = queries
            logger.info(f"Using {len(queries)} provided queries")
        elif checkpoint and checkpoint.stage >= EvaluationStage.QUERIES_GENERATED:
            self._queries = self._checkpoint_mgr.load_queries()
            logger.info(f"Resumed {len(self._queries)} queries from checkpoint")
        elif not self._queries:
            await self.generate_queries()
            # Save checkpoint
            self._checkpoint_mgr.save_queries(self._queries)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.QUERIES_GENERATED,
                total_queries=len(self._queries),
                queries_file=str(self._checkpoint_mgr.output_dir / "queries.json"),
            )

        # Step 2: Collect or load responses (supports incremental updates)
        existing_responses = None
        if checkpoint and checkpoint.stage >= EvaluationStage.RESPONSES_COLLECTED:
            existing_responses = self._checkpoint_mgr.load_responses()
            logger.info(f"Resumed {len(existing_responses)} responses from checkpoint")

        if existing_responses:
            # Check if we need to collect new endpoints
            existing_endpoints = set()
            for resp_data in existing_responses:
                existing_endpoints.update(resp_data.get("responses", {}).keys())

            config_endpoints = set(self.config.target_endpoints.keys())
            missing_endpoints = config_endpoints - existing_endpoints

            if missing_endpoints:
                logger.info(f"New endpoints detected: {missing_endpoints}, collecting incrementally...")
                await self.collect_responses(existing_responses=existing_responses)
                # Save merged responses
                self._checkpoint_mgr.save_responses(self._responses)
                self._checkpoint_mgr.update_stage(
                    EvaluationStage.RESPONSES_COLLECTED,
                    collected_responses=len(self._responses),
                    responses_file=str(self._checkpoint_mgr.output_dir / "responses.json"),
                )
            else:
                self._responses = existing_responses
        elif not self._responses:
            await self.collect_responses()
            # Save checkpoint
            self._checkpoint_mgr.save_responses(self._responses)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.RESPONSES_COLLECTED,
                collected_responses=len(self._responses),
                responses_file=str(self._checkpoint_mgr.output_dir / "responses.json"),
            )

        # Step 3: Generate or load rubrics
        if rubrics:
            self._rubrics = rubrics
            logger.info(f"Using {len(rubrics)} provided rubrics")
        elif checkpoint and checkpoint.stage >= EvaluationStage.RUBRICS_GENERATED:
            self._rubrics = self._checkpoint_mgr.load_rubrics()
            logger.info(f"Resumed {len(self._rubrics)} rubrics from checkpoint")
        elif not self._rubrics:
            await self.generate_rubrics()
            # Save checkpoint with judge model info
            self._checkpoint_mgr.save_rubrics(self._rubrics)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.RUBRICS_GENERATED,
                rubrics_file=str(self._checkpoint_mgr.output_dir / "rubrics.json"),
                judge_model=self.config.judge_endpoint.model,
                judge_base_url=self.config.judge_endpoint.base_url,
            )

        # Step 4: Run pairwise evaluation (supports incremental)
        logger.info("Step 4: Running pairwise evaluation...")
        dataset, endpoint_names = self._prepare_pairwise_data(self._responses)

        if not dataset:
            raise ValueError("No valid comparison pairs. Check if responses were collected successfully.")

        # Load existing comparison details for incremental evaluation
        existing_details = self._checkpoint_mgr.load_comparison_details()
        dataset_to_run, existing_results = self._filter_existing_comparisons(dataset, existing_details)

        if dataset_to_run:
            logger.info(
                f"Incremental evaluation: {len(existing_results)} existing, "
                f"{len(dataset_to_run)} new comparisons to run"
            )
            new_results, new_details = await self._run_pairwise_evaluation(dataset_to_run, self._rubrics)
            # Merge and deduplicate details
            self._comparison_details = self._deduplicate_comparison_details(existing_details + new_details)
            # Rebuild grader_results from all details for analysis
            grader_results = self._rebuild_grader_results(dataset, self._comparison_details)
        else:
            logger.info(f"All {len(existing_details)} comparisons already completed, skipping evaluation")
            self._comparison_details = self._deduplicate_comparison_details(existing_details)
            grader_results = existing_results

        # Save deduplicated comparison details
        self._checkpoint_mgr.save_comparison_details(self._comparison_details)

        # Step 5: Analyze results using OpenJudge's PairwiseAnalyzer
        logger.info("Step 5: Analyzing results...")
        result = self._analyze_results(dataset, grader_results, endpoint_names)

        # Mark evaluation complete and save judge model info
        self._checkpoint_mgr.update_stage(
            EvaluationStage.EVALUATION_COMPLETE,
            total_pairs=len(dataset),
            evaluated_pairs=len(grader_results),
            judge_model=self.config.judge_endpoint.model,
            judge_base_url=self.config.judge_endpoint.base_url,
        )

        self._display_results(result)

        # Step 6: Generate report if enabled
        if self.config.report.enabled:
            await self._generate_and_save_report(result)

        # Step 7: Generate win rate chart if enabled (requires report.enabled)
        if self.config.report.enabled and self.config.report.chart.enabled:
            self._generate_win_rate_chart(result)

        return result

    async def _generate_and_save_report(self, result: EvaluationResult) -> None:
        """Generate and save evaluation report."""
        from cookbooks.zero_shot_evaluation.report_generator import ReportGenerator

        logger.info("Step 6: Generating evaluation report...")
        generator = ReportGenerator(
            judge_endpoint=self.config.judge_endpoint,
            language=self.config.report.language,
            include_examples=self.config.report.include_examples,
        )
        report = await generator.generate(
            task_config=self.config.task,
            rubrics=self._rubrics,
            result=result,
            details=self._comparison_details,
        )

        # Save report
        output_dir = Path(self.config.output.output_dir)
        report_path = output_dir / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    def _generate_win_rate_chart(self, result: EvaluationResult) -> None:
        """Generate and save win rate comparison chart."""
        logger.info("Step 7: Generating win rate chart...")

        chart_config = self.config.report.chart
        generator = WinRateChartGenerator(config=chart_config)

        chart_path = generator.generate(
            rankings=result.rankings,
            output_dir=self.config.output.output_dir,
            task_description=self.config.task.description,
            total_queries=result.total_queries,
            total_comparisons=result.total_comparisons,
        )

        # Generate matrix heatmap if enabled
        if chart_config.matrix_enabled:
            logger.info("Generating win rate matrix...")
            model_order = [name for name, _ in result.rankings]
            generator.generate_matrix(
                win_matrix=result.win_matrix,
                model_order=model_order,
                output_dir=self.config.output.output_dir,
                task_description=self.config.task.description,
                total_queries=result.total_queries,
                total_comparisons=result.total_comparisons,
            )

    def _display_results(self, result: EvaluationResult) -> None:
        """Display evaluation results with formatted output."""
        endpoint_names = list(self.config.target_endpoints.keys())

        # Header
        logger.info("\n" + "=" * 60)
        logger.info("ZERO-SHOT EVALUATION RESULTS")
        logger.info("=" * 60)

        # Summary
        logger.info(f"Task: {self.config.task.description[:50]}...")
        logger.info(f"Queries: {result.total_queries}")
        logger.info(f"Comparisons: {result.total_comparisons}")

        # Rankings
        logger.info("\nRankings:")
        for rank, (name, win_rate) in enumerate(result.rankings, 1):
            bar_len = int(win_rate * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            logger.info(f"  {rank}. {name:<20} [{bar}] {win_rate:.1%}")

        # Win Matrix
        if len(endpoint_names) > 1:
            logger.info("\nWin Matrix (row vs column):")
            # Header row
            max_name_len = max(len(n) for n in endpoint_names)
            header = " " * (max_name_len + 3) + "".join(f"{n[:8]:<10}" for n in endpoint_names)
            logger.info(f"  {header}")

            # Data rows
            for ep_a in endpoint_names:
                row = f"  {ep_a:<{max_name_len}} | "
                for ep_b in endpoint_names:
                    if ep_a == ep_b:
                        row += f"{'--':<10}"
                    else:
                        win_rate = result.win_matrix.get(ep_a, {}).get(ep_b, 0.0)
                        row += f"{win_rate:<10.1%}"
                logger.info(row)

        # Best pipeline
        logger.info(f"\nBest Pipeline: {result.best_pipeline}")
        logger.info("=" * 60)

    def save_results(
        self,
        result: EvaluationResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save evaluation results to file.

        Args:
            result: Evaluation result
            output_dir: Output directory (uses config default if not provided)

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir or self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "evaluation_results.json"

        data = {
            "result": result.model_dump(),
            "config": {
                "task": self.config.task.model_dump(),
                "target_endpoints": list(self.config.target_endpoints.keys()),
                "num_queries": self.config.query_generation.num_queries,
            },
            "queries": [q.model_dump() for q in self._queries],
            "rubrics": self._rubrics,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def clear_checkpoint(self) -> None:
        """Clear all checkpoint data to start fresh."""
        self._checkpoint_mgr.clear()
