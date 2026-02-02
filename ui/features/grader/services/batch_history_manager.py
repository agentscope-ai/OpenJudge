# -*- coding: utf-8 -*-
"""History manager for Batch Grader Evaluation tasks.

Provides functionality to:
- List past batch evaluation tasks
- Load task details and results
- Resume incomplete tasks
- Delete old tasks
- Export results

Note: This manager now uses workspace-based paths for multi-user isolation.
"""

import csv
import io
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class BatchTaskSummary:
    """Summary of a batch evaluation task."""

    task_id: str
    task_dir: str
    created_at: datetime
    grader_name: str
    grader_name_zh: str
    total_count: int
    completed_count: int
    success_count: int
    failed_count: int
    status: str  # 'running', 'paused', 'completed', 'failed'
    avg_score: float | None = None
    pass_rate: float | None = None


def _get_workspace_batch_dir() -> Path:
    """Get the batch evaluations directory for the current workspace.

    Returns:
        Path to workspace-specific batch evaluations directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "batch_evaluations"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "batch_evaluations"


class BatchHistoryManager:
    """Manager for batch evaluation task history.

    Scans the batch evaluation output directory for past tasks and
    provides access to their results and details.

    Uses workspace-based paths for multi-user isolation.
    """

    # File names
    CONFIG_FILE = "config.json"
    CHECKPOINT_FILE = "checkpoint.json"
    INPUT_DATA_FILE = "input_data.json"
    RESULTS_FILE = "results.json"
    SUMMARY_FILE = "summary.json"

    def __init__(self, base_dir: Path | str | None = None):
        """Initialize history manager.

        Args:
            base_dir: Base directory for batch evaluations. If None, uses workspace directory.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = _get_workspace_batch_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_task_id(self) -> str:
        """Generate a unique task ID.

        Returns:
            Task ID in format 'batch_YYYYMMDD_HHMMSS_xxx'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds to avoid collision within same second
        micro = datetime.now().strftime("%f")[:3]
        return f"batch_{timestamp}_{micro}"

    def get_task_dir(self, task_id: str) -> Path:
        """Get the directory path for a task.

        Args:
            task_id: Task ID

        Returns:
            Path to task directory
        """
        return self.base_dir / task_id

    def create_task_dir(self, task_id: str) -> Path:
        """Create a new task directory.

        Args:
            task_id: Task ID

        Returns:
            Path to created task directory
        """
        task_dir = self.get_task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def list_tasks(self, limit: int = 20) -> list[BatchTaskSummary]:
        """List past batch evaluation tasks.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of BatchTaskSummary sorted by creation time (newest first)
        """
        tasks = []

        if not self.base_dir.exists():
            return tasks

        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if not task_dir.name.startswith("batch_"):
                continue

            summary = self._load_task_summary(task_dir)
            if summary:
                tasks.append(summary)

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def _load_task_summary(self, task_dir: Path) -> BatchTaskSummary | None:
        """Load task summary from a task directory.

        Args:
            task_dir: Path to task directory

        Returns:
            BatchTaskSummary or None if invalid
        """
        try:
            task_id = task_dir.name

            # Default values
            created_at = datetime.fromtimestamp(task_dir.stat().st_mtime)
            grader_name = "Unknown"
            grader_name_zh = "未知"
            total_count = 0
            completed_count = 0
            success_count = 0
            failed_count = 0
            status = "unknown"
            avg_score = None
            pass_rate = None

            # Load from config
            config_path = task_dir / self.CONFIG_FILE
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                grader_name = config.get("grader_name", grader_name)
                grader_name_zh = config.get("grader_name_zh", grader_name)
                created_at_str = config.get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)

            # Load from checkpoint for progress info
            checkpoint_path = task_dir / self.CHECKPOINT_FILE
            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                total_count = checkpoint.get("total_count", 0)
                completed_count = len(checkpoint.get("completed_indices", []))
                success_count = checkpoint.get("success_count", 0)
                failed_count = checkpoint.get("failed_count", 0)
                status = checkpoint.get("status", "unknown")

            # Load from summary for final stats
            summary_path = task_dir / self.SUMMARY_FILE
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                total_count = summary.get("total_count", total_count)
                completed_count = summary.get("completed_count", completed_count)
                success_count = summary.get("success_count", success_count)
                failed_count = summary.get("failed_count", failed_count)
                avg_score = summary.get("avg_score")
                pass_rate = summary.get("pass_rate")
                if summary.get("status"):
                    status = summary["status"]

            # Determine status if not explicitly set
            if status == "unknown":
                if completed_count == total_count and total_count > 0:
                    status = "completed"
                elif completed_count > 0:
                    status = "paused"
                else:
                    status = "pending"

            return BatchTaskSummary(
                task_id=task_id,
                task_dir=str(task_dir),
                created_at=created_at,
                grader_name=grader_name,
                grader_name_zh=grader_name_zh,
                total_count=total_count,
                completed_count=completed_count,
                success_count=success_count,
                failed_count=failed_count,
                status=status,
                avg_score=avg_score,
                pass_rate=pass_rate,
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

        details: dict[str, Any] = {
            "task_id": task_id,
            "task_dir": str(task_dir),
        }

        # Load config
        config_path = task_dir / self.CONFIG_FILE
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                details["config"] = json.load(f)

        # Load checkpoint
        checkpoint_path = task_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                details["checkpoint"] = json.load(f)

        # Load summary
        summary_path = task_dir / self.SUMMARY_FILE
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                details["summary"] = json.load(f)

        # Load results (may be large)
        results_path = task_dir / self.RESULTS_FILE
        details["has_results"] = results_path.exists()

        return details

    def get_task_results(
        self,
        task_id: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get evaluation results for a task.

        Args:
            task_id: Task ID
            offset: Start index for pagination
            limit: Maximum number of results to return

        Returns:
            List of result records or None if not found
        """
        task_dir = self.base_dir / task_id
        results_path = task_dir / self.RESULTS_FILE

        if not results_path.exists():
            return None

        try:
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)

            if not isinstance(all_results, list):
                return None

            if limit is None:
                return all_results[offset:]
            return all_results[offset : offset + limit]

        except Exception as e:
            logger.error(f"Failed to load results for {task_id}: {e}")
            return None

    def get_task_input_data(self, task_id: str) -> list[dict[str, Any]] | None:
        """Get original input data for a task.

        Args:
            task_id: Task ID

        Returns:
            List of input records or None if not found
        """
        task_dir = self.base_dir / task_id
        input_path = task_dir / self.INPUT_DATA_FILE

        if not input_path.exists():
            return None

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input data for {task_id}: {e}")
            return None

    def save_config(self, task_id: str, config: dict[str, Any]) -> bool:
        """Save task configuration.

        Args:
            task_id: Task ID
            config: Configuration dict

        Returns:
            True if saved successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)

            config_path = task_dir / self.CONFIG_FILE
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save config for {task_id}: {e}")
            return False

    def save_input_data(self, task_id: str, data: list[dict[str, Any]]) -> bool:
        """Save input data for a task.

        Args:
            task_id: Task ID
            data: Input data list

        Returns:
            True if saved successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)

            input_path = task_dir / self.INPUT_DATA_FILE
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save input data for {task_id}: {e}")
            return False

    def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> bool:
        """Save checkpoint for a task.

        Args:
            task_id: Task ID
            checkpoint: Checkpoint dict

        Returns:
            True if saved successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            checkpoint_path = task_dir / self.CHECKPOINT_FILE

            # Write to temp file first, then rename (atomic operation)
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            temp_path.replace(checkpoint_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {task_id}: {e}")
            return False

    def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """Load checkpoint for a task.

        Args:
            task_id: Task ID

        Returns:
            Checkpoint dict or None if not found
        """
        task_dir = self.get_task_dir(task_id)
        checkpoint_path = task_dir / self.CHECKPOINT_FILE

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {task_id}: {e}")
            return None

    def save_results(self, task_id: str, results: list[dict[str, Any]]) -> bool:
        """Save evaluation results for a task.

        Args:
            task_id: Task ID
            results: List of result records

        Returns:
            True if saved successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            results_path = task_dir / self.RESULTS_FILE

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save results for {task_id}: {e}")
            return False

    def append_result(self, task_id: str, result: dict[str, Any]) -> bool:
        """Append a single result to the results file.

        Uses append mode to avoid loading entire file.

        Args:
            task_id: Task ID
            result: Single result record

        Returns:
            True if appended successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            results_path = task_dir / self.RESULTS_FILE

            # If file doesn't exist, create with array start
            if not results_path.exists():
                with open(results_path, "w", encoding="utf-8") as f:
                    f.write("[\n")
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n]")
            else:
                # Read file, remove closing bracket, append new result
                with open(results_path, "r+", encoding="utf-8") as f:
                    f.seek(0, 2)  # Go to end
                    pos = f.tell()
                    # Find the last ']' and replace it
                    f.seek(max(0, pos - 10))
                    content = f.read()
                    bracket_pos = content.rfind("]")
                    if bracket_pos >= 0:
                        f.seek(max(0, pos - 10) + bracket_pos)
                        f.write(",\n")
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n]")
                        f.truncate()

            return True
        except Exception as e:
            logger.error(f"Failed to append result for {task_id}: {e}")
            return False

    def save_summary(self, task_id: str, summary: dict[str, Any]) -> bool:
        """Save evaluation summary for a task.

        Args:
            task_id: Task ID
            summary: Summary statistics dict

        Returns:
            True if saved successfully
        """
        try:
            task_dir = self.get_task_dir(task_id)
            summary_path = task_dir / self.SUMMARY_FILE

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save summary for {task_id}: {e}")
            return False

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and all its files.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deleted successfully
        """
        task_dir = self.base_dir / task_id
        if not task_dir.exists():
            return False

        try:
            shutil.rmtree(task_dir)
            logger.info(f"Deleted batch task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False

    def export_results(
        self,
        task_id: str,
        format_type: str = "json",
    ) -> bytes | None:
        """Export task results in specified format.

        Args:
            task_id: Task ID
            format_type: Export format ('json' or 'csv')

        Returns:
            Exported data as bytes or None
        """
        results = self.get_task_results(task_id)
        if not results:
            return None

        details = self.get_task_details(task_id)

        if format_type == "json":
            export_data = {
                "task_id": task_id,
                "config": details.get("config", {}) if details else {},
                "summary": details.get("summary", {}) if details else {},
                "results": results,
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8")

        elif format_type == "csv":
            if not results:
                return None

            output = io.StringIO()

            # Determine fieldnames from first result
            # Standard fields + result fields
            fieldnames = ["index", "status", "score", "passed", "reason"]

            # collect all input keys
            all_input_keys = set()
            for result in results:
                input_data = result.get("input", {})
                if isinstance(input_data, dict):
                    all_input_keys.update(input_data.keys())

            for key in all_input_keys:
                if key not in fieldnames:
                    fieldnames.append(f"input_{key}")

            writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for result in results:
                row = {
                    "index": result.get("index", ""),
                    "status": result.get("status", ""),
                    "score": result.get("score", ""),
                    "passed": result.get("passed", ""),
                    "reason": result.get("reason", "")[:500] if result.get("reason") else "",
                }
                # Add input fields
                input_data = result.get("input", {})
                for key, value in input_data.items():
                    if isinstance(value, (dict, list)):
                        row[f"input_{key}"] = json.dumps(value, ensure_ascii=False)[:500]
                    else:
                        row[f"input_{key}"] = str(value)[:500] if value else ""

                writer.writerow(row)

            return output.getvalue().encode("utf-8")

        return None

    def get_incomplete_tasks(self) -> list[BatchTaskSummary]:
        """Get list of incomplete tasks that can be resumed.

        Returns:
            List of incomplete tasks
        """
        all_tasks = self.list_tasks(limit=100)
        return [
            task
            for task in all_tasks
            if task.status in ("running", "paused") and task.completed_count < task.total_count
        ]

    def cleanup_old_tasks(self, keep_days: int = 30, keep_count: int = 50) -> int:
        """Clean up old completed tasks to save disk space.

        Args:
            keep_days: Keep tasks newer than this many days
            keep_count: Always keep at least this many tasks

        Returns:
            Number of tasks deleted
        """
        all_tasks = self.list_tasks(limit=1000)
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

        # Sort by date, keep recent ones
        tasks_to_check = all_tasks[keep_count:]

        deleted_count = 0
        for task in tasks_to_check:
            if task.created_at.timestamp() < cutoff_date and task.status == "completed":
                if self.delete_task(task.task_id):
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old batch evaluation tasks")

        return deleted_count
