# -*- coding: utf-8 -*-
"""History service for Paper Review feature.

Provides persistent storage for review history using JSON files.

Note: This manager now uses workspace-based paths for multi-user isolation.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from features.paper_review.services.pipeline_runner import ReviewTaskResult
from loguru import logger


def _get_workspace_history_dir() -> Path:
    """Get the paper review history directory for the current workspace.

    Returns:
        Path to workspace-specific paper review history directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "paper_review_history"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "paper_review_history"


@dataclass
class HistoryEntry:
    """A single history entry."""

    task_id: str
    paper_name: str
    created_at: str
    success: bool
    elapsed_seconds: float
    review_score: Optional[int] = None
    correctness_score: Optional[int] = None
    report: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_result(
        cls,
        result: ReviewTaskResult,
        task_id: Optional[str] = None,
    ) -> "HistoryEntry":
        """Create history entry from task result.

        Args:
            result: The review task result
            task_id: Optional task ID (generates new if not provided)

        Returns:
            HistoryEntry instance
        """
        review_score = None
        correctness_score = None

        if result.result:
            if result.result.review:
                review_score = result.result.review.score
            if result.result.correctness:
                # Invert for display (1=best -> 3, 3=worst -> 1)
                correctness_score = 4 - result.result.correctness.score

        return cls(
            task_id=task_id or str(uuid.uuid4()),
            paper_name=result.paper_name,
            created_at=datetime.now().isoformat(),
            success=result.success,
            elapsed_seconds=result.elapsed_seconds,
            review_score=review_score,
            correctness_score=correctness_score,
            report=result.report,
            error=result.error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryEntry":
        """Create from dictionary."""
        return cls(**data)

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

    @property
    def created_at_display(self) -> str:
        """Format created time for display."""
        try:
            dt = datetime.fromisoformat(self.created_at)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return self.created_at


class HistoryService:
    """Service for managing paper review history.

    Uses workspace-based paths for multi-user isolation.
    """

    def __init__(self, history_dir: Optional[Path] = None):
        """Initialize history service.

        Args:
            history_dir: Directory to store history files. If None, uses workspace directory.
        """
        if history_dir:
            self.history_dir = history_dir
        else:
            self.history_dir = _get_workspace_history_dir()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure history directory exists."""
        os.makedirs(self.history_dir, exist_ok=True)

    def _get_entry_path(self, task_id: str) -> Path:
        """Get file path for a history entry."""
        return self.history_dir / f"{task_id}.json"

    def _get_index_path(self) -> Path:
        """Get path to the index file."""
        return self.history_dir / "index.json"

    def _load_index(self) -> list[dict[str, Any]]:
        """Load the history index."""
        index_path = self._get_index_path()
        if not index_path.exists():
            return []

        try:
            with open(index_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load history index: {e}")
            return []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        """Save the history index."""
        index_path = self._get_index_path()
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to save history index: {e}")

    def save(self, result: ReviewTaskResult) -> str:
        """Save a review result to history.

        Args:
            result: The review task result to save

        Returns:
            Task ID of the saved entry
        """
        # Create entry
        task_id = str(uuid.uuid4())
        entry = HistoryEntry.from_task_result(result, task_id)

        # Save entry file
        entry_path = self._get_entry_path(task_id)
        try:
            with open(entry_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to save history entry: {e}")
            raise

        # Update index
        index = self._load_index()
        index.insert(
            0,
            {
                "task_id": task_id,
                "paper_name": entry.paper_name,
                "created_at": entry.created_at,
                "success": entry.success,
                "review_score": entry.review_score,
            },
        )

        # Keep only last 100 entries in index
        if len(index) > 100:
            # Remove old entries
            old_entries = index[100:]
            index = index[:100]
            for old in old_entries:
                self.delete(old["task_id"])

        self._save_index(index)

        logger.info(f"Saved review history: {task_id}")
        return task_id

    def get(self, task_id: str) -> Optional[HistoryEntry]:
        """Get a history entry by ID.

        Args:
            task_id: The task ID

        Returns:
            HistoryEntry or None if not found
        """
        entry_path = self._get_entry_path(task_id)
        if not entry_path.exists():
            return None

        try:
            with open(entry_path, encoding="utf-8") as f:
                data = json.load(f)
                return HistoryEntry.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load history entry {task_id}: {e}")
            return None

    def list_all(self, limit: int = 50) -> list[HistoryEntry]:
        """List all history entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of history entries (newest first)
        """
        index = self._load_index()
        entries: list[HistoryEntry] = []

        for item in index[:limit]:
            entry = self.get(item["task_id"])
            if entry:
                entries.append(entry)

        return entries

    def delete(self, task_id: str) -> bool:
        """Delete a history entry.

        Args:
            task_id: The task ID to delete

        Returns:
            True if deleted, False if not found
        """
        entry_path = self._get_entry_path(task_id)

        # Remove file
        try:
            if entry_path.exists():
                os.remove(entry_path)
        except OSError as e:
            logger.warning(f"Failed to delete history file {task_id}: {e}")

        # Update index
        index = self._load_index()
        original_len = len(index)
        index = [item for item in index if item["task_id"] != task_id]

        if len(index) < original_len:
            self._save_index(index)
            logger.info(f"Deleted history entry: {task_id}")
            return True

        return False

    def clear_all(self) -> int:
        """Clear all history entries.

        Returns:
            Number of entries deleted
        """
        index = self._load_index()
        count = len(index)

        for item in index:
            entry_path = self._get_entry_path(item["task_id"])
            try:
                if entry_path.exists():
                    os.remove(entry_path)
            except OSError:
                pass

        self._save_index([])
        logger.info(f"Cleared {count} history entries")
        return count

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for history.

        Returns:
            Dictionary with summary stats
        """
        entries = self.list_all(limit=100)

        if not entries:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "avg_review_score": None,
                "avg_correctness_score": None,
            }

        success = sum(1 for e in entries if e.success)
        failed = len(entries) - success

        review_scores = [e.review_score for e in entries if e.review_score is not None]
        correctness_scores = [e.correctness_score for e in entries if e.correctness_score is not None]

        return {
            "total": len(entries),
            "success": success,
            "failed": failed,
            "avg_review_score": sum(review_scores) / len(review_scores) if review_scores else None,
            "avg_correctness_score": sum(correctness_scores) / len(correctness_scores) if correctness_scores else None,
        }


def get_history_service() -> HistoryService:
    """Get the history service instance for the current workspace.

    Note: This creates a new instance each time to ensure it uses the
    current workspace path. The workspace manager handles caching of
    the workspace path itself, so this is still efficient.
    """
    return HistoryService()
