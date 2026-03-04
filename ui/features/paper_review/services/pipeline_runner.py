# -*- coding: utf-8 -*-
"""Pipeline runner service for Paper Review feature.

Provides a wrapper around PaperReviewPipeline with progress tracking
and error handling suitable for UI integration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

from cookbooks.paper_review import (
    PaperReviewPipeline,
    PaperReviewResult,
    PipelineConfig,
    ReviewProgress,
    generate_report,
)


@dataclass
class ReviewTaskConfig:
    """Configuration for a paper review task."""

    # API settings
    api_key: str
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    temperature: float = 0.7

    # Review options
    enable_safety_checks: bool = True
    enable_correctness: bool = True
    enable_review: bool = True
    enable_criticality: bool = True
    enable_bib_verification: bool = True
    crossref_mailto: Optional[str] = None

    # Paper info
    paper_name: str = "Paper"

    def to_pipeline_config(
        self,
        progress_callback: Optional[Callable[[ReviewProgress], None]] = None,
    ) -> PipelineConfig:
        """Convert to PipelineConfig."""
        return PipelineConfig(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            enable_safety_checks=self.enable_safety_checks,
            enable_correctness=self.enable_correctness,
            enable_review=self.enable_review,
            enable_criticality=self.enable_criticality,
            enable_bib_verification=self.enable_bib_verification,
            crossref_mailto=self.crossref_mailto,
            progress_callback=progress_callback,
        )


@dataclass
class ReviewTaskResult:
    """Result of a paper review task."""

    success: bool
    result: Optional[PaperReviewResult] = None
    report: Optional[str] = None
    error: Optional[str] = None
    paper_name: str = "Paper"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0

    @property
    def elapsed_time_display(self) -> str:
        """Format elapsed time for display."""
        if self.elapsed_seconds < 1:
            return f"{self.elapsed_seconds * 1000:.0f}ms"
        elif self.elapsed_seconds < 60:
            return f"{self.elapsed_seconds:.1f}s"
        else:
            minutes = int(self.elapsed_seconds // 60)
            secs = self.elapsed_seconds % 60
            return f"{minutes}m {secs:.0f}s"


class PipelineRunner:
    """Runner for paper review pipeline with UI integration."""

    def __init__(
        self,
        config: ReviewTaskConfig,
        progress_callback: Optional[Callable[[ReviewProgress], None]] = None,
    ):
        """Initialize pipeline runner.

        Args:
            config: Review task configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self._progress_callback = progress_callback
        self._current_progress: Optional[ReviewProgress] = None

    def _on_progress(self, progress: ReviewProgress) -> None:
        """Handle progress update from pipeline."""
        self._current_progress = progress
        if self._progress_callback:
            self._progress_callback(progress)

    async def run_async(
        self,
        pdf_bytes: bytes,
        bib_content: Optional[str] = None,
    ) -> ReviewTaskResult:
        """Run paper review asynchronously.

        Args:
            pdf_bytes: PDF file content as bytes
            bib_content: Optional .bib file content

        Returns:
            ReviewTaskResult with results or error
        """
        started_at = datetime.now()

        try:
            # Create pipeline with progress callback
            pipeline_config = self.config.to_pipeline_config(progress_callback=self._on_progress)
            pipeline = PaperReviewPipeline(pipeline_config)

            # Run review
            result = await pipeline.review_paper(
                pdf_input=pdf_bytes,
                bib_content=bib_content,
            )

            completed_at = datetime.now()
            elapsed = (completed_at - started_at).total_seconds()

            # Check if safety check failed (pipeline returns early with is_safe=False)
            if not result.is_safe:
                return ReviewTaskResult(
                    success=False,
                    result=result,
                    error="Safety check failed: " + "; ".join(result.safety_issues),
                    paper_name=self.config.paper_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    elapsed_seconds=elapsed,
                )

            # Generate report for successful review
            report = generate_report(result, self.config.paper_name)

            return ReviewTaskResult(
                success=True,
                result=result,
                report=report,
                paper_name=self.config.paper_name,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            completed_at = datetime.now()
            elapsed = (completed_at - started_at).total_seconds()

            return ReviewTaskResult(
                success=False,
                error=str(e),
                paper_name=self.config.paper_name,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
            )

    def run(
        self,
        pdf_bytes: bytes,
        bib_content: Optional[str] = None,
    ) -> ReviewTaskResult:
        """Run paper review synchronously.

        Args:
            pdf_bytes: PDF file content as bytes
            bib_content: Optional .bib file content

        Returns:
            ReviewTaskResult with results or error
        """
        from shared.utils.helpers import run_async

        return run_async(self.run_async(pdf_bytes, bib_content))

    @property
    def current_progress(self) -> Optional[ReviewProgress]:
        """Get current progress state."""
        return self._current_progress


def create_task_config_from_sidebar(sidebar_config: dict[str, Any]) -> ReviewTaskConfig:
    """Create ReviewTaskConfig from sidebar configuration.

    Args:
        sidebar_config: Configuration dictionary from sidebar

    Returns:
        ReviewTaskConfig instance
    """
    return ReviewTaskConfig(
        api_key=sidebar_config.get("api_key", ""),
        model_name=sidebar_config.get("model_name", "gpt-4o"),
        base_url=sidebar_config.get("api_endpoint"),
        temperature=sidebar_config.get("temperature", 0.7),
        enable_safety_checks=sidebar_config.get("enable_safety_checks", True),
        enable_correctness=sidebar_config.get("enable_correctness", True),
        enable_review=sidebar_config.get("enable_review", True),
        enable_criticality=sidebar_config.get("enable_criticality", True),
        enable_bib_verification=sidebar_config.get("enable_bib_verification", True),
        crossref_mailto=sidebar_config.get("crossref_mailto"),
    )
