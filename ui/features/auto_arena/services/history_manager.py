# -*- coding: utf-8 -*-
"""History manager for Auto Arena tasks.

Provides functionality to:
- List past evaluation tasks
- Load task details and results
- Delete old tasks

Note: This manager now uses workspace-based paths for multi-user isolation.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TaskSummary:
    """Summary of a past evaluation task."""

    task_id: str
    task_dir: str
    created_at: datetime
    task_description: str
    target_models: list[str]
    num_queries: int
    status: str  # 'completed', 'failed', 'in_progress'
    best_model: str | None = None
    win_rates: dict[str, float] | None = None


def _get_workspace_evaluations_dir() -> Path:
    """Get the evaluations directory for the current workspace.

    Returns:
        Path to workspace-specific evaluations directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "evaluations"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "evaluations"


class HistoryManager:
    """Manager for evaluation task history.

    Scans the evaluation output directory for past tasks and
    provides access to their results and details.

    Uses workspace-based paths for multi-user isolation.
    """

    CHECKPOINT_FILE = "checkpoint.json"
    RESULTS_FILE = "evaluation_results.json"
    REPORT_FILE = "evaluation_report.md"
    CHART_FILE = "win_rate_chart.png"

    def __init__(self, base_dir: Path | str | None = None):
        """Initialize history manager.

        Args:
            base_dir: Base directory for evaluations. If None, uses workspace directory.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = _get_workspace_evaluations_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_tasks(self, limit: int = 20) -> list[TaskSummary]:
        """List past evaluation tasks.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of TaskSummary sorted by creation time (newest first)
        """
        tasks = []

        if not self.base_dir.exists():
            return tasks

        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue

            summary = self._load_task_summary(task_dir)
            if summary:
                tasks.append(summary)

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def _load_task_summary(self, task_dir: Path) -> TaskSummary | None:
        """Load task summary from a task directory.

        Args:
            task_dir: Path to task directory

        Returns:
            TaskSummary or None if invalid
        """
        try:
            task_id = task_dir.name

            # Try to load checkpoint first
            checkpoint_path = task_dir / self.CHECKPOINT_FILE
            results_path = task_dir / self.RESULTS_FILE

            created_at = datetime.fromtimestamp(task_dir.stat().st_mtime)
            task_description = ""
            target_models = []
            num_queries = 0
            status = "in_progress"
            best_model = None
            win_rates = None

            # Load from checkpoint
            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                created_at = datetime.fromisoformat(checkpoint.get("created_at", created_at.isoformat()))
                status = checkpoint.get("stage", "in_progress")
                num_queries = checkpoint.get("total_queries", 0)

                if status == "evaluation_complete":
                    status = "completed"

            # Load from results
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)

                config = results.get("config", {})
                task_description = config.get("task", {}).get("description", "")
                target_models = config.get("target_endpoints", [])
                num_queries = config.get("num_queries", num_queries)

                result = results.get("result", {})
                best_model = result.get("best_pipeline")
                win_rates = result.get("win_rates")

                if best_model:
                    status = "completed"

            return TaskSummary(
                task_id=task_id,
                task_dir=str(task_dir),
                created_at=created_at,
                task_description=task_description[:100] if task_description else "Unknown task",
                target_models=target_models if isinstance(target_models, list) else list(target_models),
                num_queries=num_queries,
                status=status,
                best_model=best_model,
                win_rates=win_rates,
            )

        except Exception as e:
            logger.warning(f"Failed to load task summary from {task_dir}: {e}")
            return None

    def get_task_details(self, task_id: str) -> dict[str, Any] | None:
        """Get full details of a task.

        Args:
            task_id: Task ID (directory name)

        Returns:
            Full task details dict or None
        """
        task_dir = self.base_dir / task_id
        if not task_dir.exists():
            return None

        details = {
            "task_id": task_id,
            "task_dir": str(task_dir),
        }

        # Load results
        results_path = task_dir / self.RESULTS_FILE
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                details["results"] = json.load(f)

        # Load queries
        queries_path = task_dir / "queries.json"
        if queries_path.exists():
            with open(queries_path, "r", encoding="utf-8") as f:
                details["queries"] = json.load(f)

        # Load comparison details
        details_path = task_dir / "comparison_details.json"
        if details_path.exists():
            with open(details_path, "r", encoding="utf-8") as f:
                details["comparison_details"] = json.load(f)

        # Check for report and chart
        details["has_report"] = (task_dir / self.REPORT_FILE).exists()
        details["has_chart"] = (task_dir / self.CHART_FILE).exists()

        return details

    def get_report_content(self, task_id: str) -> str | None:
        """Get the report content for a task.

        Args:
            task_id: Task ID

        Returns:
            Report markdown content or None
        """
        report_path = self.base_dir / task_id / self.REPORT_FILE
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def get_chart_path(self, task_id: str) -> str | None:
        """Get the chart file path for a task.

        Args:
            task_id: Task ID

        Returns:
            Chart file path or None
        """
        chart_path = self.base_dir / task_id / self.CHART_FILE
        if chart_path.exists():
            return str(chart_path)
        return None

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and all its files.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deleted successfully
        """
        import shutil

        task_dir = self.base_dir / task_id
        if not task_dir.exists():
            return False

        try:
            shutil.rmtree(task_dir)
            logger.info(f"Deleted task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False

    def export_task(self, task_id: str, format: str = "json") -> bytes | None:  # pylint: disable=redefined-builtin
        """Export task data in specified format.

        Args:
            task_id: Task ID
            format: Export format ('json' or 'csv')

        Returns:
            Exported data as bytes or None
        """
        details = self.get_task_details(task_id)
        if not details:
            return None

        if format == "json":
            return json.dumps(details, indent=2, ensure_ascii=False).encode("utf-8")

        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write rankings as CSV
            results = details.get("results", {}).get("result", {})
            rankings = results.get("rankings", [])

            writer.writerow(["Rank", "Model", "Win Rate"])
            for rank, (name, win_rate) in enumerate(rankings, 1):
                writer.writerow([rank, name, f"{win_rate:.2%}"])

            return output.getvalue().encode("utf-8")

        return None
