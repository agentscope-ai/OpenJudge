# -*- coding: utf-8 -*-
"""Pipeline runner service for Auto Arena.

Wraps the AutoArenaPipeline with progress tracking and UI integration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from loguru import logger


class PipelineStage(str, Enum):
    """Pipeline stage identifiers."""

    NOT_STARTED = "not_started"
    QUERIES = "queries"
    RESPONSES = "responses"
    RUBRICS = "rubrics"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PipelineProgress:
    """Progress state for the pipeline."""

    stage: PipelineStage = PipelineStage.NOT_STARTED
    total_progress: float = 0.0
    stage_progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_dir: str = ""

    def add_log(self, message: str) -> None:
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def update_stage(
        self,
        stage: PipelineStage,
        progress: float = 0.0,
        message: str = "",
    ) -> None:
        """Update stage progress."""
        self.stage = stage
        self.stage_progress[stage.value] = {
            "progress": progress,
            "message": message,
        }
        self._update_total_progress()

    def _update_total_progress(self) -> None:
        """Calculate total progress based on stage.

        Each stage has a start and end weight. Progress within a stage
        interpolates between these weights based on stage_progress (0-1).
        """
        # Ordered stages with their START weights (when stage begins)
        # The end weight is the start weight of the next stage
        stage_weights = [
            (PipelineStage.NOT_STARTED, 0.0),
            (PipelineStage.QUERIES, 0.0),  # 0% - 15%
            (PipelineStage.RESPONSES, 0.15),  # 15% - 45%
            (PipelineStage.RUBRICS, 0.45),  # 45% - 55%
            (PipelineStage.EVALUATION, 0.55),  # 55% - 90%
            (PipelineStage.ANALYSIS, 0.90),  # 90% - 100%
            (PipelineStage.COMPLETED, 1.0),
        ]

        # Handle terminal states
        if self.stage == PipelineStage.COMPLETED:
            self.total_progress = 1.0
            return
        if self.stage in [PipelineStage.FAILED, PipelineStage.PAUSED]:
            # Keep current progress for failed/paused states
            return

        # Find current stage index
        current_idx = -1
        for i, (stage, _) in enumerate(stage_weights):
            if stage == self.stage:
                current_idx = i
                break

        if current_idx < 0 or current_idx == 0:
            self.total_progress = 0.0
            return

        # Get stage boundaries (start weight of current stage, start weight of next stage)
        start_weight = stage_weights[current_idx][1]
        end_weight = stage_weights[current_idx + 1][1] if current_idx + 1 < len(stage_weights) else 1.0

        # Calculate progress within current stage
        stage_progress = self.stage_progress.get(self.stage.value, {}).get("progress", 0.0)
        stage_range = end_weight - start_weight
        self.total_progress = start_weight + (stage_range * stage_progress)


class PipelineRunner:
    """Runner for Auto Arena evaluation pipeline with progress tracking.

    Provides:
    - Async pipeline execution
    - Progress callbacks for UI updates
    - Pause/resume capability (via checkpoint system)
    - Error handling
    """

    def __init__(
        self,
        config: dict[str, Any],
        progress_callback: Callable[[PipelineProgress], None] | None = None,
        resume_from: str | None = None,
    ):
        """Initialize pipeline runner.

        Args:
            config: Pipeline configuration dict
            progress_callback: Optional callback for progress updates
            resume_from: Optional output_dir to resume from
        """
        self.config = config
        self.progress_callback = progress_callback
        self.progress = PipelineProgress()
        self._is_paused = False
        self._is_cancelled = False
        self._pipeline = None
        self._resume_from = resume_from

    CONFIG_FILE = "ui_config.json"  # Saved UI config for resume

    @classmethod
    def resume(
        cls,
        output_dir: str,
        progress_callback: Callable[[PipelineProgress], None] | None = None,
    ) -> "PipelineRunner":
        """Create a runner to resume a paused evaluation.

        Args:
            output_dir: Directory of the paused evaluation
            progress_callback: Optional callback for progress updates

        Returns:
            PipelineRunner configured for resume

        Raises:
            FileNotFoundError: If no config can be loaded
        """
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        config_path = output_path / cls.CONFIG_FILE
        results_path = output_path / "evaluation_results.json"
        checkpoint_path = output_path / "checkpoint.json"

        config = {}

        # Priority: ui_config.json > evaluation_results.json
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded UI config from {config_path}")
        elif results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = data.get("config", {})
            logger.info("Loaded config from results file")

        if not config:
            raise FileNotFoundError(
                f"No configuration found in {output_dir}. " "Cannot resume evaluation without original config."
            )

        runner = cls(config, progress_callback, resume_from=output_dir)
        runner.config["output_dir"] = output_dir
        runner.progress.output_dir = output_dir

        # Set initial progress based on checkpoint
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            stage_map = {
                "not_started": PipelineStage.NOT_STARTED,
                "queries_generated": PipelineStage.QUERIES,
                "responses_collected": PipelineStage.RESPONSES,
                "rubrics_generated": PipelineStage.RUBRICS,
                "evaluation_complete": PipelineStage.COMPLETED,
            }
            checkpoint_stage = checkpoint.get("stage", "not_started")
            runner.progress.stage = stage_map.get(checkpoint_stage, PipelineStage.NOT_STARTED)
            runner.progress.add_log(f"Resuming from checkpoint: {checkpoint_stage}")

        return runner

    def save_config(self) -> None:
        """Save UI config for resume capability."""
        import json
        from pathlib import Path

        output_dir = self.config.get("output_dir")
        if not output_dir:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / self.CONFIG_FILE

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved UI config to {config_path}")

    def _notify_progress(self) -> None:
        """Notify progress callback if set."""
        if self.progress_callback:
            self.progress_callback(self.progress)

    def _build_arena_config(self) -> Any:
        """Build AutoArenaConfig from UI config dict."""
        from cookbooks.auto_arena.schema import (
            AutoArenaConfig,
            ChartConfig,
            EvaluationConfig,
            OpenAIEndpoint,
            OutputConfig,
            QueryGenerationConfig,
            ReportConfig,
            TaskConfig,
        )

        # Task config
        task = TaskConfig(
            description=self.config.get("task_description", ""),
            scenario=self.config.get("task_scenario"),
        )

        # Target endpoints
        target_endpoints = {}
        for name, ep_config in self.config.get("target_endpoints", {}).items():
            if ep_config.get("api_key") and ep_config.get("model"):
                target_endpoints[name] = OpenAIEndpoint(
                    base_url=ep_config.get("base_url", ""),
                    api_key=ep_config.get("api_key", ""),
                    model=ep_config.get("model", ""),
                    system_prompt=ep_config.get("system_prompt"),
                )

        # Judge endpoint
        judge_endpoint = OpenAIEndpoint(
            base_url=self.config.get("judge_endpoint", ""),
            api_key=self.config.get("judge_api_key", ""),
            model=self.config.get("judge_model", ""),
        )

        # Query generation
        query_generation = QueryGenerationConfig(
            num_queries=self.config.get("num_queries", 20),
            seed_queries=self.config.get("seed_queries") or None,
            temperature=self.config.get("query_temperature", 0.9),
            max_similarity=self.config.get("max_similarity", 0.85),
            enable_evolution=self.config.get("enable_evolution", False),
        )

        # Evaluation config
        evaluation = EvaluationConfig(
            max_concurrency=self.config.get("max_concurrency", 10),
        )

        # Output config
        output_dir = self.config.get("output_dir", "./evaluation_results")
        output = OutputConfig(
            save_queries=self.config.get("save_queries", True),
            save_responses=self.config.get("save_responses", True),
            save_details=self.config.get("save_details", True),
            output_dir=output_dir,
        )
        self.progress.output_dir = output_dir

        # Report config
        report = ReportConfig(
            enabled=self.config.get("generate_report", True),
            chart=ChartConfig(enabled=self.config.get("generate_chart", True)),
        )

        return AutoArenaConfig(
            task=task,
            target_endpoints=target_endpoints,
            judge_endpoint=judge_endpoint,
            query_generation=query_generation,
            evaluation=evaluation,
            output=output,
            report=report,
        )

    def validate_config(self) -> tuple[bool, str]:  # pylint: disable=too-many-return-statements
        """Validate the configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check task description
        if not self.config.get("task_description"):
            return False, "Task description is required"

        # Check target endpoints
        endpoints = self.config.get("target_endpoints", {})
        if len(endpoints) < 2:
            return False, "At least 2 target models are required for comparison"

        for name, ep in endpoints.items():
            if not ep.get("api_key"):
                return False, f"API key missing for target model '{name}'"
            if not ep.get("model"):
                return False, f"Model name missing for target model '{name}'"

        # Check judge endpoint
        if not self.config.get("judge_api_key"):
            return False, "Judge model API key is required"
        if not self.config.get("judge_model"):
            return False, "Judge model name is required"

        return True, ""

    async def run(self) -> dict[str, Any] | None:  # pylint: disable=too-many-return-statements,too-many-statements
        """Run the evaluation pipeline.

        Returns:
            Evaluation result dict or None if failed/cancelled
        """
        from cookbooks.auto_arena.auto_arena_pipeline import AutoArenaPipeline

        self.progress = PipelineProgress()
        self.progress.started_at = datetime.now()
        self._is_paused = False
        self._is_cancelled = False

        try:
            # Validate config
            is_valid, error = self.validate_config()
            if not is_valid:
                self.progress.error = error
                self.progress.stage = PipelineStage.FAILED
                self._notify_progress()
                return None

            self.progress.add_log("Initializing pipeline...")
            self._notify_progress()

            # Build config
            arena_config = self._build_arena_config()

            # Create pipeline
            self._pipeline = AutoArenaPipeline(config=arena_config, resume=True)

            # Stage 1: Generate queries
            self.progress.update_stage(PipelineStage.QUERIES, 0.0, "Generating test queries...")
            self.progress.add_log("Stage 1: Generating test queries")
            self._notify_progress()

            if self._is_cancelled:
                return None

            queries = await self._pipeline.generate_queries()
            self.progress.update_stage(PipelineStage.QUERIES, 1.0, f"Generated {len(queries)} queries")
            self.progress.add_log(f"Generated {len(queries)} test queries")
            self._notify_progress()

            # Stage 2: Collect responses
            self.progress.update_stage(PipelineStage.RESPONSES, 0.0, "Collecting model responses...")
            self.progress.add_log("Stage 2: Collecting responses from target models")
            self._notify_progress()

            if self._is_cancelled:
                return None

            responses = await self._pipeline.collect_responses()
            self.progress.update_stage(PipelineStage.RESPONSES, 1.0, f"Collected {len(responses)} responses")
            self.progress.add_log(f"Collected {len(responses)} responses")
            self._notify_progress()

            # Stage 3: Generate rubrics
            self.progress.update_stage(PipelineStage.RUBRICS, 0.0, "Generating evaluation rubrics...")
            self.progress.add_log("Stage 3: Generating evaluation rubrics")
            self._notify_progress()

            if self._is_cancelled:
                return None

            rubrics = await self._pipeline.generate_rubrics()
            self.progress.update_stage(PipelineStage.RUBRICS, 1.0, f"Generated {len(rubrics)} rubrics")
            self.progress.add_log(f"Generated {len(rubrics)} evaluation rubrics")
            self._notify_progress()

            # Stage 4: Run evaluation
            self.progress.update_stage(PipelineStage.EVALUATION, 0.0, "Running pairwise evaluation...")
            self.progress.add_log("Stage 4: Running pairwise evaluation")
            self._notify_progress()

            if self._is_cancelled:
                return None

            # Note: The actual evaluate() method handles stages 4-5
            # pylint: disable=protected-access
            result = await self._pipeline.evaluate(
                queries=self._pipeline._queries,
                rubrics=self._pipeline._rubrics,
            )
            # pylint: enable=protected-access

            # Stage 5: Analysis complete
            self.progress.update_stage(PipelineStage.ANALYSIS, 1.0, "Analysis complete")
            self.progress.add_log("Stage 5: Analysis complete")

            # Completed
            self.progress.stage = PipelineStage.COMPLETED
            self.progress.total_progress = 1.0
            self.progress.completed_at = datetime.now()
            self.progress.result = result.model_dump()
            self.progress.add_log(f"Evaluation completed! Best model: {result.best_pipeline}")
            self._notify_progress()

            # Save results
            self._pipeline.save_results(result)

            return result.model_dump()

        except asyncio.CancelledError:
            self.progress.stage = PipelineStage.PAUSED
            self.progress.add_log("Pipeline cancelled")
            self._notify_progress()
            return None

        except Exception as e:
            logger.exception("Pipeline failed")
            self.progress.stage = PipelineStage.FAILED
            self.progress.error = str(e)
            self.progress.add_log(f"Error: {e}")
            self._notify_progress()
            return None

    def pause(self) -> None:
        """Pause the pipeline (will pause at next checkpoint)."""
        self._is_paused = True
        self.progress.add_log("Pause requested...")
        self._notify_progress()

    def cancel(self) -> None:
        """Cancel the pipeline."""
        self._is_cancelled = True
        self.progress.add_log("Cancellation requested...")
        self._notify_progress()

    def get_progress(self) -> PipelineProgress:
        """Get current progress state."""
        return self.progress
