# -*- coding: utf-8 -*-
"""Batch runner service for Paper Review feature.

Provides batch processing capabilities for reviewing multiple papers.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from features.paper_review.services.pipeline_runner import (
    PipelineRunner,
    ReviewTaskConfig,
    ReviewTaskResult,
)
from loguru import logger


class BatchStatus(str, Enum):
    """Status of a batch review task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchPaperItem:
    """A single paper item in a batch."""

    paper_id: str
    paper_name: str
    pdf_bytes: bytes
    bib_content: Optional[str] = None
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[ReviewTaskResult] = None


@dataclass
class BatchProgress:
    """Progress state for batch review."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    current_paper: Optional[str] = None
    current_paper_progress: float = 0.0
    status: BatchStatus = BatchStatus.PENDING

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.total == 0:
            return 0.0
        base_progress = (self.completed / self.total) * 100
        if self.current_paper:
            paper_contribution = self.current_paper_progress / self.total
            return base_progress + paper_contribution
        return base_progress

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "current_paper": self.current_paper,
            "current_paper_progress": self.current_paper_progress,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
        }


@dataclass
class BatchResult:
    """Result of a batch review."""

    success: bool
    results: list[ReviewTaskResult] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    failed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0

    @property
    def elapsed_time_display(self) -> str:
        """Format elapsed time for display."""
        if self.elapsed_seconds < 60:
            return f"{self.elapsed_seconds:.1f}s"
        minutes = int(self.elapsed_seconds // 60)
        secs = self.elapsed_seconds % 60
        return f"{minutes}m {secs:.0f}s"


BatchProgressCallback = Callable[[BatchProgress], None]


class BatchRunner:
    """Runner for batch paper review."""

    def __init__(
        self,
        config: ReviewTaskConfig,
        progress_callback: Optional[BatchProgressCallback] = None,
    ):
        """Initialize batch runner.

        Args:
            config: Base review task configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self._progress_callback = progress_callback
        self._progress = BatchProgress()
        self._cancelled = False

    def _notify_progress(self) -> None:
        """Notify progress callback."""
        if self._progress_callback:
            self._progress_callback(self._progress)

    async def run_async(
        self,
        papers: list[BatchPaperItem],
        max_concurrency: int = 1,
    ) -> BatchResult:
        """Run batch review asynchronously.

        Args:
            papers: List of papers to review
            max_concurrency: Maximum concurrent reviews (default 1 for stability)

        Returns:
            BatchResult with all review results
        """
        started_at = datetime.now()
        self._cancelled = False

        # Initialize progress
        self._progress = BatchProgress(
            total=len(papers),
            status=BatchStatus.RUNNING,
        )
        self._notify_progress()

        results: list[ReviewTaskResult] = []

        try:
            # Process papers with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrency)

            async def process_paper(paper: BatchPaperItem) -> ReviewTaskResult:
                async with semaphore:
                    if self._cancelled:
                        return ReviewTaskResult(
                            success=False,
                            error="Cancelled",
                            paper_name=paper.paper_name,
                        )

                    # Update progress
                    self._progress.current_paper = paper.paper_name
                    self._progress.current_paper_progress = 0.0
                    self._notify_progress()

                    # Create config for this paper
                    paper_config = ReviewTaskConfig(
                        api_key=self.config.api_key,
                        model_name=self.config.model_name,
                        base_url=self.config.base_url,
                        temperature=self.config.temperature,
                        enable_safety_checks=self.config.enable_safety_checks,
                        enable_correctness=self.config.enable_correctness,
                        enable_review=self.config.enable_review,
                        enable_criticality=self.config.enable_criticality,
                        enable_bib_verification=self.config.enable_bib_verification,
                        crossref_mailto=self.config.crossref_mailto,
                        paper_name=paper.paper_name,
                    )

                    # Progress callback for individual paper
                    def on_paper_progress(progress) -> None:
                        self._progress.current_paper_progress = progress.progress_percent
                        self._notify_progress()

                    # Run review
                    runner = PipelineRunner(paper_config, progress_callback=on_paper_progress)
                    result = await runner.run_async(paper.pdf_bytes, paper.bib_content)

                    # Update progress
                    if result.success:
                        self._progress.completed += 1
                    else:
                        self._progress.failed += 1

                    self._progress.current_paper = None
                    self._progress.current_paper_progress = 0.0
                    self._notify_progress()

                    return result

            # Run all papers
            tasks = [process_paper(paper) for paper in papers]
            results = await asyncio.gather(*tasks)

            # Finalize
            completed_at = datetime.now()
            elapsed = (completed_at - started_at).total_seconds()

            self._progress.status = BatchStatus.COMPLETED
            self._notify_progress()

            return BatchResult(
                success=True,
                results=list(results),
                total=len(papers),
                completed=self._progress.completed,
                failed=self._progress.failed,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            logger.exception(f"Batch review failed: {e}")
            completed_at = datetime.now()
            elapsed = (completed_at - started_at).total_seconds()

            self._progress.status = BatchStatus.FAILED
            self._notify_progress()

            return BatchResult(
                success=False,
                results=results,
                total=len(papers),
                completed=self._progress.completed,
                failed=self._progress.failed + 1,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
            )

    def run(
        self,
        papers: list[BatchPaperItem],
        max_concurrency: int = 1,
    ) -> BatchResult:
        """Run batch review synchronously.

        Args:
            papers: List of papers to review
            max_concurrency: Maximum concurrent reviews

        Returns:
            BatchResult with all review results
        """
        from shared.utils.helpers import run_async

        return run_async(self.run_async(papers, max_concurrency))

    def cancel(self) -> None:
        """Cancel the batch review."""
        self._cancelled = True
        self._progress.status = BatchStatus.CANCELLED
        self._notify_progress()

    @property
    def progress(self) -> BatchProgress:
        """Get current progress."""
        return self._progress


def generate_batch_csv_report(results: list[ReviewTaskResult]) -> str:
    """Generate a CSV summary report for batch results.

    Args:
        results: List of review task results

    Returns:
        CSV formatted string
    """
    lines = ["Paper Name,Status,Review Score,Correctness Score,Elapsed Time,Error"]

    for result in results:
        status = "Success" if result.success else "Failed"
        review_score = ""
        correctness_score = ""

        if result.result:
            if result.result.review:
                review_score = str(result.result.review.score)
            if result.result.correctness:
                # Invert score for display
                correctness_score = str(4 - result.result.correctness.score)

        error = result.error.replace(",", ";") if result.error else ""
        elapsed = result.elapsed_time_display

        lines.append(f'"{result.paper_name}",{status},{review_score},{correctness_score},{elapsed},"{error}"')

    return "\n".join(lines)
