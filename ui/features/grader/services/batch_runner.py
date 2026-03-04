# -*- coding: utf-8 -*-
"""Batch evaluation runner for Grader feature.

Provides:
- Concurrent batch evaluation
- Checkpoint-based resume capability
- Progress tracking with callbacks
- Error handling per item
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from features.grader.services.batch_history_manager import BatchHistoryManager
from features.grader.services.grader_factory import (
    create_grader,
    run_agent_evaluation,
    run_evaluation,
)
from loguru import logger
from shared.services.model_factory import create_model

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


class BatchStatus(str, Enum):
    """Batch evaluation status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchProgress:
    """Progress state for batch evaluation."""

    task_id: str = ""
    status: BatchStatus = BatchStatus.PENDING
    total_count: int = 0
    completed_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    current_index: int = -1
    started_at: datetime | None = None
    elapsed_seconds: float = 0.0
    avg_time_per_item: float = 0.0
    estimated_remaining_seconds: float = 0.0
    error: str | None = None
    logs: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_count == 0:
            return 0.0
        return (self.completed_count / self.total_count) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "current_index": self.current_index,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_time_per_item": self.avg_time_per_item,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "error": self.error,
        }


@dataclass
class BatchResult:
    """Single evaluation result."""

    index: int
    status: str  # 'success' or 'error'
    score: float | None = None
    passed: bool | None = None
    reason: str = ""
    error: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "status": self.status,
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "error": self.error,
            "input": self.input_data,
            "metadata": self.metadata,
            "elapsed_ms": self.elapsed_ms,
        }


class BatchRunner:
    """Runner for batch grader evaluation.

    Handles concurrent evaluation with checkpoint-based resume capability.
    """

    # Checkpoint save frequency (save every N completed items)
    CHECKPOINT_FREQUENCY = 10

    def __init__(
        self,
        task_id: str,
        grader_name: str,
        grader_config: dict[str, Any],
        api_config: dict[str, Any],
        data: list[dict[str, Any]],
        max_concurrency: int = 10,
        history_manager: BatchHistoryManager | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ):
        """Initialize batch runner.

        Args:
            task_id: Unique task identifier
            grader_name: Name of the grader to use
            grader_config: Grader configuration from registry
            api_config: API configuration (endpoint, key, model, etc.)
            data: List of input data records
            max_concurrency: Maximum concurrent evaluations
            history_manager: History manager for persistence
            progress_callback: Optional callback for progress updates
        """
        self.task_id = task_id
        self.grader_name = grader_name
        self.grader_config = grader_config
        self.api_config = api_config
        self.data = data
        self.max_concurrency = max(1, max_concurrency)
        self.history_manager = history_manager or BatchHistoryManager()
        self.progress_callback = progress_callback

        # State
        self.progress = BatchProgress(
            task_id=task_id,
            total_count=len(data),
        )
        self.results: list[BatchResult] = []
        self.completed_indices: set[int] = set()
        self._is_cancelled = False
        self._grader: BaseGrader | None = None
        self._model: OpenAIChatModel | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def resume(
        cls,
        task_id: str,
        api_config: dict[str, Any],
        history_manager: BatchHistoryManager | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> "BatchRunner | None":
        """Create a runner to resume an incomplete task.

        Args:
            task_id: Task ID to resume
            api_config: Current API configuration (must include api_key from sidebar)
            history_manager: History manager instance
            progress_callback: Progress callback

        Returns:
            BatchRunner instance or None if task cannot be resumed
        """
        manager = history_manager or BatchHistoryManager()

        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        if not checkpoint:
            logger.error(f"No checkpoint found for task {task_id}")
            return None

        # Load config
        details = manager.get_task_details(task_id)
        if not details or "config" not in details:
            logger.error(f"No config found for task {task_id}")
            return None

        config = details["config"]

        # Load input data
        input_data = manager.get_task_input_data(task_id)
        if not input_data:
            logger.error(f"No input data found for task {task_id}")
            return None

        # Merge saved api_config with provided api_config (provided takes precedence)
        # This ensures we use the current API key while preserving other saved settings
        saved_api_config = config.get("api_config", {})
        merged_api_config = {**saved_api_config, **api_config}

        # Create runner
        runner = cls(
            task_id=task_id,
            grader_name=config.get("grader_name", ""),
            grader_config=config.get("grader_config", {}),
            api_config=merged_api_config,
            data=input_data,
            max_concurrency=config.get("max_concurrency", 10),
            history_manager=manager,
            progress_callback=progress_callback,
        )

        # Restore state from checkpoint
        runner.completed_indices = set(checkpoint.get("completed_indices", []))
        runner.progress.completed_count = len(runner.completed_indices)
        runner.progress.success_count = checkpoint.get("success_count", 0)
        runner.progress.failed_count = checkpoint.get("failed_count", 0)

        # Load existing results
        existing_results = manager.get_task_results(task_id)
        if existing_results:
            for result_dict in existing_results:
                runner.results.append(
                    BatchResult(
                        index=result_dict.get("index", 0),
                        status=result_dict.get("status", "error"),
                        score=result_dict.get("score"),
                        passed=result_dict.get("passed"),
                        reason=result_dict.get("reason", ""),
                        error=result_dict.get("error"),
                        input_data=result_dict.get("input", {}),
                        metadata=result_dict.get("metadata", {}),
                        elapsed_ms=result_dict.get("elapsed_ms", 0),
                    )
                )

        runner.progress.add_log(f"Resuming task from checkpoint: {len(runner.completed_indices)} items completed")

        return runner

    def _notify_progress(self) -> None:
        """Notify progress callback if set."""
        if self.progress_callback:
            self.progress_callback(self.progress)

    def _create_model(self) -> OpenAIChatModel | None:
        """Create model instance if required by grader."""
        if not self.grader_config.get("requires_model", False):
            return None

        return create_model(
            api_key=self.api_config.get("api_key", ""),
            base_url=self.api_config.get("api_endpoint"),
            model_name=self.api_config.get("model_name", ""),
        )

    def _create_grader(self, model: OpenAIChatModel | None) -> BaseGrader:
        """Create grader instance."""
        language = self.api_config.get("language", LanguageEnum.EN)
        # Handle language being a string (e.g., from saved config)
        if isinstance(language, str):
            # Check for various string representations of ZH
            lang_upper = language.upper()
            is_zh = "ZH" in lang_upper or "中文" in language
            language = LanguageEnum.ZH if is_zh else LanguageEnum.EN

        threshold = self.api_config.get("threshold", 0.5)
        extra_params = self.api_config.get("extra_params", {})

        return create_grader(
            grader_name=self.grader_name,
            model=model,
            threshold=threshold,
            language=language,
            **extra_params,
        )

    async def _evaluate_single(
        self,
        index: int,
        input_data: dict[str, Any],
        grader: BaseGrader,
        semaphore: asyncio.Semaphore,
    ) -> BatchResult:
        """Evaluate a single item.

        Args:
            index: Item index
            input_data: Input data for evaluation
            grader: Grader instance
            semaphore: Concurrency semaphore

        Returns:
            BatchResult with evaluation result
        """
        async with semaphore:
            if self._is_cancelled:
                return BatchResult(
                    index=index,
                    status="error",
                    error="Cancelled",
                    input_data=input_data,
                )

            start_time = time.time()

            try:
                category = self.grader_config.get("category", "common")

                if category == "agent":
                    # Agent evaluation
                    result = run_agent_evaluation(
                        grader=grader,
                        query=input_data.get("query", ""),
                        tool_definitions=input_data.get("tool_definitions", []),
                        tool_calls=input_data.get("tool_calls", []),
                        reference_tool_calls=input_data.get("reference_tool_calls"),
                    )
                else:
                    # Standard evaluation
                    result = run_evaluation(
                        grader=grader,
                        query=input_data.get("query", ""),
                        response=input_data.get("response", ""),
                        reference_response=input_data.get("reference_response", ""),
                        context=input_data.get("context", ""),
                    )

                elapsed_ms = (time.time() - start_time) * 1000

                if isinstance(result, GraderError):
                    return BatchResult(
                        index=index,
                        status="error",
                        error=result.error,
                        reason=result.reason,
                        input_data=input_data,
                        elapsed_ms=elapsed_ms,
                    )

                # GraderScore
                threshold = self.api_config.get("threshold", 0.5)
                passed = result.score >= threshold

                return BatchResult(
                    index=index,
                    status="success",
                    score=result.score,
                    passed=passed,
                    reason=result.reason,
                    input_data=input_data,
                    metadata=result.metadata or {},
                    elapsed_ms=elapsed_ms,
                )

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.exception(f"Error evaluating item {index}")
                return BatchResult(
                    index=index,
                    status="error",
                    error=str(e),
                    input_data=input_data,
                    elapsed_ms=elapsed_ms,
                )

    async def _process_result(self, result: BatchResult) -> None:
        """Process a completed result.

        Updates progress, saves checkpoint periodically.

        Args:
            result: Completed batch result
        """
        async with self._lock:
            self.results.append(result)
            self.completed_indices.add(result.index)

            if result.status == "success":
                self.progress.success_count += 1
            else:
                self.progress.failed_count += 1

            self.progress.completed_count = len(self.completed_indices)
            self.progress.current_index = result.index

            # Update timing estimates
            elapsed = (datetime.now() - self.progress.started_at).total_seconds() if self.progress.started_at else 0
            self.progress.elapsed_seconds = elapsed

            if self.progress.completed_count > 0:
                self.progress.avg_time_per_item = elapsed / self.progress.completed_count
                remaining = self.progress.total_count - self.progress.completed_count
                self.progress.estimated_remaining_seconds = remaining * self.progress.avg_time_per_item

            # Log progress periodically
            if self.progress.completed_count % 10 == 0 or self.progress.completed_count == self.progress.total_count:
                self.progress.add_log(
                    f"Progress: {self.progress.completed_count}/{self.progress.total_count} "
                    f"(Success: {self.progress.success_count}, Failed: {self.progress.failed_count})"
                )

            # Save checkpoint periodically
            if self.progress.completed_count % self.CHECKPOINT_FREQUENCY == 0:
                self._save_checkpoint()

            self._notify_progress()

    def _save_checkpoint(self) -> None:
        """Save checkpoint for resume capability."""
        checkpoint = {
            "task_id": self.task_id,
            "status": self.progress.status.value,
            "total_count": self.progress.total_count,
            "completed_indices": list(self.completed_indices),
            "success_count": self.progress.success_count,
            "failed_count": self.progress.failed_count,
            "updated_at": datetime.now().isoformat(),
        }
        self.history_manager.save_checkpoint(self.task_id, checkpoint)

        # Also save results incrementally
        results_data = [r.to_dict() for r in self.results]
        self.history_manager.save_results(self.task_id, results_data)

    def _save_config(self) -> None:
        """Save task configuration."""
        config = {
            "task_id": self.task_id,
            "grader_name": self.grader_name,
            "grader_name_zh": self.grader_config.get("name_zh", self.grader_name),
            "grader_config": self.grader_config,
            "api_config": {
                # Don't save API key for security
                "api_endpoint": self.api_config.get("api_endpoint"),
                "model_name": self.api_config.get("model_name"),
                "threshold": self.api_config.get("threshold"),
                "language": str(self.api_config.get("language", "EN")),
                "extra_params": self.api_config.get("extra_params", {}),
            },
            "max_concurrency": self.max_concurrency,
            "total_count": len(self.data),
            "created_at": datetime.now().isoformat(),
        }
        self.history_manager.save_config(self.task_id, config)

    def _save_summary(self) -> None:
        """Save final summary statistics."""
        # Calculate statistics
        scores = [r.score for r in self.results if r.status == "success" and r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        passed_count = sum(1 for r in self.results if r.passed is True)
        pass_rate = passed_count / len(scores) if scores else None

        summary = {
            "task_id": self.task_id,
            "status": self.progress.status.value,
            "total_count": self.progress.total_count,
            "completed_count": self.progress.completed_count,
            "success_count": self.progress.success_count,
            "failed_count": self.progress.failed_count,
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "elapsed_seconds": self.progress.elapsed_seconds,
            "avg_time_per_item": self.progress.avg_time_per_item,
            "completed_at": datetime.now().isoformat(),
        }
        self.history_manager.save_summary(self.task_id, summary)

    async def run(self) -> BatchProgress:
        """Run the batch evaluation.

        Returns:
            Final BatchProgress with results
        """
        self.progress.status = BatchStatus.RUNNING
        self.progress.started_at = datetime.now()
        self.progress.add_log(f"Starting batch evaluation: {self.progress.total_count} items")
        self._notify_progress()

        try:
            # Save initial config and input data
            self._save_config()
            self.history_manager.save_input_data(self.task_id, self.data)

            # Create model and grader
            self.progress.add_log("Creating model and grader...")
            self._model = self._create_model()
            self._grader = self._create_grader(self._model)
            self.progress.add_log(f"Using grader: {self.grader_name}")

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrency)
            self.progress.add_log(f"Max concurrency: {self.max_concurrency}")

            # Create tasks for items not yet completed
            tasks = []
            for i, item in enumerate(self.data):
                if i not in self.completed_indices:
                    task = asyncio.create_task(self._evaluate_single(i, item, self._grader, semaphore))
                    tasks.append(task)

            # Process results as they complete
            for coro in asyncio.as_completed(tasks):
                if self._is_cancelled:
                    break
                result = await coro
                await self._process_result(result)

            # Final status
            if self._is_cancelled:
                self.progress.status = BatchStatus.PAUSED
                self.progress.add_log("Evaluation paused by user")
            elif self.progress.failed_count == self.progress.total_count:
                self.progress.status = BatchStatus.FAILED
                self.progress.add_log("Evaluation failed: all items failed")
            else:
                self.progress.status = BatchStatus.COMPLETED
                self.progress.add_log(
                    f"Evaluation completed: {self.progress.success_count} success, "
                    f"{self.progress.failed_count} failed"
                )

        except Exception as e:
            logger.exception("Batch evaluation failed")
            self.progress.status = BatchStatus.FAILED
            self.progress.error = str(e)
            self.progress.add_log(f"Error: {e}")

        finally:
            # Save final checkpoint and summary
            self._save_checkpoint()
            self._save_summary()
            self._notify_progress()

        return self.progress

    def cancel(self) -> None:
        """Cancel the running evaluation."""
        self._is_cancelled = True
        self.progress.add_log("Cancellation requested...")
        self._notify_progress()

    def get_results(self) -> list[dict[str, Any]]:
        """Get all results as dictionaries.

        Returns:
            List of result dictionaries
        """
        return [r.to_dict() for r in sorted(self.results, key=lambda r: r.index)]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Summary dictionary
        """
        scores = [r.score for r in self.results if r.status == "success" and r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        passed_count = sum(1 for r in self.results if r.passed is True)
        pass_rate = passed_count / len(scores) if scores else None

        return {
            "total_count": self.progress.total_count,
            "completed_count": self.progress.completed_count,
            "success_count": self.progress.success_count,
            "failed_count": self.progress.failed_count,
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "elapsed_seconds": self.progress.elapsed_seconds,
        }
