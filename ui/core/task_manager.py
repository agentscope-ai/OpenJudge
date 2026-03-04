# -*- coding: utf-8 -*-
"""Task manager for background task management in OpenJudge Studio.

This module provides a TaskManager for managing long-running tasks like
Auto Arena evaluation, AutoRubric generation, etc. It supports:
- Task creation and tracking
- Progress updates
- Pause/resume functionality
- State persistence across page refreshes
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Paused by user
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user


class TaskType(str, Enum):
    """Type of task."""

    AUTO_ARENA = "auto_arena"
    AUTORUBRIC = "autorubric"
    PAIRWISE_EVAL = "pairwise_eval"
    BATCH_GRADING = "batch_grading"


@dataclass
class TaskProgress:
    """Progress information for a task."""

    current_stage: str = ""
    stage_progress: dict[str, Any] = field(default_factory=dict)
    total_progress: float = 0.0
    message: str = ""
    logs: list[str] = field(default_factory=list)


@dataclass
class TaskInfo:
    """Information about a task."""

    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    updated_at: datetime

    # Configuration snapshot
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""

    # Progress information
    progress: TaskProgress = field(default_factory=TaskProgress)

    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Error information
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "config_snapshot": self.config_snapshot,
            "output_dir": self.output_dir,
            "progress": {
                "current_stage": self.progress.current_stage,
                "stage_progress": self.progress.stage_progress,
                "total_progress": self.progress.total_progress,
                "message": self.progress.message,
                "logs": self.progress.logs[-100:],  # Keep last 100 logs
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInfo":
        """Create from dictionary."""
        progress_data = data.get("progress", {})
        progress = TaskProgress(
            current_stage=progress_data.get("current_stage", ""),
            stage_progress=progress_data.get("stage_progress", {}),
            total_progress=progress_data.get("total_progress", 0.0),
            message=progress_data.get("message", ""),
            logs=progress_data.get("logs", []),
        )

        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            config_snapshot=data.get("config_snapshot", {}),
            output_dir=data.get("output_dir", ""),
            progress=progress,
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            error_message=data.get("error_message"),
        )


class TaskManager:
    """Manager for background tasks.

    Stores task information in the file system for persistence across
    page refreshes and application restarts.

    Storage location: ~/.openjudge_studio/tasks/
    """

    DEFAULT_TASKS_DIR = Path.home() / ".openjudge_studio" / "tasks"

    def __init__(self, tasks_dir: Optional[Path] = None):
        """Initialize task manager.

        Args:
            tasks_dir: Directory to store task files (default: ~/.openjudge_studio/tasks/)
        """
        self.tasks_dir = tasks_dir or self.DEFAULT_TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"TaskManager initialized with tasks_dir: {self.tasks_dir}")

    def _get_task_file(self, task_id: str) -> Path:
        """Get the file path for a task."""
        return self.tasks_dir / f"{task_id}.json"

    def create_task(
        self,
        task_type: TaskType,
        config: dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> TaskInfo:
        """Create a new task.

        Args:
            task_type: Type of the task
            config: Configuration snapshot for the task
            output_dir: Output directory for task results

        Returns:
            Created TaskInfo
        """
        task_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        # Default output directory
        if output_dir is None:
            output_dir = str(self.tasks_dir / "outputs" / f"{task_type.value}_{task_id}")

        # Create output directory to ensure it exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")

        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            config_snapshot=config,
            output_dir=output_dir,
        )

        self._save_task(task)
        logger.info(f"Created task: {task_id} (type={task_type.value})")

        return task

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            TaskInfo or None if not found
        """
        task_file = self._get_task_file(task_id)
        if not task_file.exists():
            return None

        try:
            with open(task_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TaskInfo.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return None

    def list_tasks(
        self,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ) -> list[TaskInfo]:
        """List tasks with optional filtering.

        Args:
            task_type: Filter by task type
            status: Filter by status
            limit: Maximum number of tasks to return

        Returns:
            List of TaskInfo objects, sorted by updated_at descending
        """
        tasks = []

        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                task = TaskInfo.from_dict(data)

                # Apply filters
                if task_type and task.task_type != task_type:
                    continue
                if status and task.status != status:
                    continue

                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to load task file {task_file}: {e}")
                continue

        # Sort by updated_at descending
        tasks.sort(key=lambda t: t.updated_at, reverse=True)

        return tasks[:limit]

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[TaskProgress] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TaskInfo]:
        """Update a task.

        Args:
            task_id: Task ID
            status: New status
            progress: New progress information
            error_message: Error message (for failed tasks)

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.get_task(task_id)
        if not task:
            return None

        task.updated_at = datetime.now()

        if status:
            task.status = status
            if status == TaskStatus.RUNNING and not task.start_time:
                task.start_time = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.end_time = datetime.now()

        if progress:
            task.progress = progress

        if error_message:
            task.error_message = error_message

        self._save_task(task)
        logger.debug(f"Updated task {task_id}: status={task.status.value}")

        return task

    def update_progress(
        self,
        task_id: str,
        stage: str,
        stage_progress: dict[str, Any],
        total_progress: float,
        message: str = "",
        log: Optional[str] = None,
    ) -> Optional[TaskInfo]:
        """Update task progress.

        Args:
            task_id: Task ID
            stage: Current stage name
            stage_progress: Progress for each stage
            total_progress: Overall progress (0-1)
            message: Current status message
            log: Optional log message to append

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.get_task(task_id)
        if not task:
            return None

        task.progress.current_stage = stage
        task.progress.stage_progress = stage_progress
        task.progress.total_progress = total_progress
        task.progress.message = message

        if log:
            timestamp = datetime.now().strftime("%H:%M:%S")
            task.progress.logs.append(f"[{timestamp}] {log}")
            # Keep only last 100 logs
            if len(task.progress.logs) > 100:
                task.progress.logs = task.progress.logs[-100:]

        task.updated_at = datetime.now()
        self._save_task(task)

        return task

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task.

        Args:
            task_id: Task ID

        Returns:
            True if paused successfully
        """
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        task.status = TaskStatus.PAUSED
        task.updated_at = datetime.now()
        self._save_task(task)

        logger.info(f"Paused task: {task_id}")
        return True

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task.

        Args:
            task_id: Task ID

        Returns:
            True if resumed successfully
        """
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.PAUSED:
            return False

        task.status = TaskStatus.RUNNING
        task.updated_at = datetime.now()
        self._save_task(task)

        logger.info(f"Resumed task: {task_id}")
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled successfully
        """
        task = self.get_task(task_id)
        if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False

        task.status = TaskStatus.CANCELLED
        task.end_time = datetime.now()
        task.updated_at = datetime.now()
        self._save_task(task)

        logger.info(f"Cancelled task: {task_id}")
        return True

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID

        Returns:
            True if deleted successfully
        """
        task_file = self._get_task_file(task_id)
        if task_file.exists():
            task_file.unlink()
            logger.info(f"Deleted task: {task_id}")
            return True
        return False

    def get_running_tasks(self, task_type: Optional[TaskType] = None) -> list[TaskInfo]:
        """Get all running or paused tasks.

        Args:
            task_type: Filter by task type

        Returns:
            List of running/paused tasks
        """
        tasks = self.list_tasks(task_type=task_type)
        return [t for t in tasks if t.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]]

    def _save_task(self, task: TaskInfo) -> None:
        """Save task to file."""
        task_file = self._get_task_file(task.task_id)
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance.

    Returns:
        TaskManager instance
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
