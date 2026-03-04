# -*- coding: utf-8 -*-
"""History manager for Auto Rubric feature.

Manages the storage and retrieval of generated graders and their configurations.

Note: This manager now uses workspace-based paths for multi-user isolation.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _get_workspace_rubrics_dir() -> Path:
    """Get the rubrics directory for the current workspace.

    Returns:
        Path to workspace-specific rubrics directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "rubrics"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "rubrics"


class HistoryManager:
    """Manager for Auto Rubric generation history.

    Stores generated graders and configurations in the workspace directory
    for later retrieval, export, and management.

    Directory structure (within workspace):
        {workspace}/rubrics/
        ├── rubric_20240115_143000/
        │   ├── config.json       # Generation configuration
        │   ├── rubrics.txt       # Generated rubrics text
        │   └── grader.json       # Grader configuration for export
        └── rubric_20240114_101500/
            └── ...
    """

    def __init__(self, base_dir: str | None = None):
        """Initialize the history manager.

        Args:
            base_dir: Base directory for storing history.
                      If None, uses workspace directory.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = _get_workspace_rubrics_dir()

        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_task_id(self) -> str:
        """Generate a unique task ID.

        Returns:
            Task ID in format: rubric_YYYYMMDD_HHMMSS
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rubric_{timestamp}"

    def save_grader(
        self,
        task_id: str,
        config: dict[str, Any],
        rubrics: str,
        grader_config: dict[str, Any],
        mode: str = "simple",
        data_count: int | None = None,
    ) -> bool:
        """Save a generated grader to history.

        Args:
            task_id: Unique task identifier.
            config: Generation configuration (input parameters).
            rubrics: Generated rubrics text.
            grader_config: Grader configuration for export.
            mode: Generation mode ("simple" or "iterative").
            data_count: Number of training data records (for iterative mode).

        Returns:
            True if save was successful.
        """
        try:
            task_dir = self.base_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration
            config_data = {
                "task_id": task_id,
                "mode": mode,
                "grader_name": config.get("grader_name", ""),
                "task_description": config.get("task_description", ""),
                "grader_mode": config.get("grader_mode", "pointwise"),
                "language": config.get("language", "EN"),
                "min_score": config.get("min_score", 0),
                "max_score": config.get("max_score", 5),
                "data_count": data_count,
                "created_at": datetime.now().isoformat(),
                "model_name": config.get("model_name", ""),
            }

            with open(task_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            # Save rubrics text
            with open(task_dir / "rubrics.txt", "w", encoding="utf-8") as f:
                f.write(rubrics)

            # Save grader configuration
            with open(task_dir / "grader.json", "w", encoding="utf-8") as f:
                json.dump(grader_config, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved grader to history: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save grader: {e}")
            return False

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        """List all saved tasks.

        Args:
            limit: Maximum number of tasks to return.

        Returns:
            List of task summaries, sorted by creation time (newest first).
        """
        tasks = []

        try:
            for task_dir in self.base_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                config_file = task_dir / "config.json"
                if not config_file.exists():
                    continue

                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    tasks.append(
                        {
                            "task_id": task_dir.name,
                            "grader_name": config.get("grader_name", ""),
                            "mode": config.get("mode", "simple"),
                            "grader_mode": config.get("grader_mode", "pointwise"),
                            "data_count": config.get("data_count"),
                            "created_at": config.get("created_at", ""),
                            "task_dir": str(task_dir),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read config for {task_dir.name}: {e}")
                    continue

            # Sort by creation time (newest first)
            tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            return tasks[:limit]

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []

    def get_task_details(self, task_id: str) -> dict[str, Any] | None:
        """Get detailed information about a task.

        Args:
            task_id: Task identifier.

        Returns:
            Task details including config, rubrics, and grader config.
        """
        task_dir = self.base_dir / task_id

        if not task_dir.exists():
            return None

        try:
            details: dict[str, Any] = {
                "task_id": task_id,
                "task_dir": str(task_dir),
            }

            # Load config
            config_file = task_dir / "config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    details["config"] = json.load(f)

            # Load rubrics
            rubrics_file = task_dir / "rubrics.txt"
            if rubrics_file.exists():
                with open(rubrics_file, "r", encoding="utf-8") as f:
                    details["rubrics"] = f.read()

            # Load grader config
            grader_file = task_dir / "grader.json"
            if grader_file.exists():
                with open(grader_file, "r", encoding="utf-8") as f:
                    details["grader_config"] = json.load(f)

            return details

        except Exception as e:
            logger.error(f"Failed to get task details: {e}")
            return None

    def delete_task(self, task_id: str) -> bool:
        """Delete a task from history.

        Args:
            task_id: Task identifier.

        Returns:
            True if deletion was successful.
        """
        task_dir = self.base_dir / task_id

        if not task_dir.exists():
            return False

        try:
            shutil.rmtree(task_dir)
            logger.info(f"Deleted task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
            return False

    def get_rubrics_count(self, rubrics: str) -> int:
        """Count the number of rubrics in the text.

        Args:
            rubrics: Rubrics text.

        Returns:
            Estimated number of rubrics.
        """
        if not rubrics:
            return 0

        # Count numbered items (1., 2., etc.) or "Rubric X:" patterns
        import re

        # Match patterns like "1.", "2." at line start or "Rubric 1:", "Rubric 2:"
        pattern = r"(?:^\d+\.|^Rubric \d+:)"
        matches = re.findall(pattern, rubrics, re.MULTILINE)
        return len(matches) if matches else 1
