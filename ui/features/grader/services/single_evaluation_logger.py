# -*- coding: utf-8 -*-
"""Logger for single (interactive) grader evaluations.

Provides functionality to log single evaluation runs for analytics and debugging.
Uses JSON Lines format for efficient append-only logging.

Note: This logger uses workspace-based paths for multi-user isolation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _get_workspace_logs_dir() -> Path:
    """Get the single evaluation logs directory for the current workspace.

    Returns:
        Path to workspace-specific single evaluation logs directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "single_evaluations"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "single_evaluations"


class SingleEvaluationLogger:
    """Logger for single evaluation runs.

    Logs evaluation data to a JSON Lines file for later analysis.
    Each line is a complete JSON object representing one evaluation.

    File structure:
        {workspace}/single_evaluations/
        └── evaluations.jsonl    # Append-only log file
    """

    LOG_FILE = "evaluations.jsonl"

    def __init__(self, base_dir: Path | str | None = None):
        """Initialize the logger.

        Args:
            base_dir: Base directory for logs. If None, uses workspace directory.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = _get_workspace_logs_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self) -> Path:
        """Get path to the log file."""
        return self.base_dir / self.LOG_FILE

    def log_evaluation(
        self,
        grader_name: str,
        input_data: dict[str, Any],
        result: Any,
        threshold: float,
        elapsed_time: float,
        model_name: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> bool:
        """Log a single evaluation run.

        Args:
            grader_name: Name of the grader used
            input_data: Input data (query, response, etc.)
            result: Evaluation result object
            threshold: Pass/fail threshold used
            elapsed_time: Time taken for evaluation in seconds
            model_name: Name of the LLM model used (if any)
            extra_params: Any extra parameters used

        Returns:
            True if logging was successful, False otherwise
        """
        try:
            # Build log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "grader_name": grader_name,
                "threshold": threshold,
                "elapsed_time_seconds": round(elapsed_time, 3),
                "model_name": model_name,
                "input": self._sanitize_input(input_data),
                "result": self._serialize_result(result),
                "extra_params": extra_params or {},
            }

            # Append to log file
            log_path = self._get_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.debug(f"Logged single evaluation: {grader_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to log single evaluation: {e}")
            return False

    def _sanitize_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize input data for logging.

        Removes internal flags and truncates very long content.

        Args:
            input_data: Raw input data

        Returns:
            Sanitized input data
        """
        sanitized = {}
        max_length = 10000  # Truncate very long content

        for key, value in input_data.items():
            # Skip internal flags
            if key.startswith("has_"):
                continue

            # Handle string values
            if isinstance(value, str):
                if len(value) > max_length:
                    sanitized[key] = value[:max_length] + "... [truncated]"
                else:
                    sanitized[key] = value
            # Handle list values (e.g., multimodal content)
            elif isinstance(value, list):
                sanitized[key] = f"[list with {len(value)} items]"
            else:
                sanitized[key] = value

        return sanitized

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize evaluation result for logging.

        Args:
            result: Evaluation result object

        Returns:
            Serialized result dict
        """
        # Handle GraderError
        if hasattr(result, "error"):
            return {
                "type": "error",
                "name": getattr(result, "name", "unknown"),
                "error": str(getattr(result, "error", "")),
                "reason": str(getattr(result, "reason", "")),
            }

        # Handle normal GraderResult
        return {
            "type": "success",
            "score": getattr(result, "score", None),
            "reason": getattr(result, "reason", ""),
            "passed": getattr(result, "score", 0) >= 0,  # Will be recalculated with threshold
            "metadata": getattr(result, "metadata", None),
        }

    def get_recent_evaluations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent evaluation logs.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of evaluation entries (newest first)
        """
        log_path = self._get_log_path()
        if not log_path.exists():
            return []

        entries = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to read evaluation logs: {e}")
            return []

        # Return newest first
        entries.reverse()
        return entries[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about logged evaluations.

        Returns:
            Statistics dict with counts by grader, success rate, etc.
        """
        entries = self.get_recent_evaluations(limit=10000)

        if not entries:
            return {"total_count": 0}

        grader_counts: dict[str, int] = {}
        success_count = 0
        error_count = 0

        for entry in entries:
            grader = entry.get("grader_name", "unknown")
            grader_counts[grader] = grader_counts.get(grader, 0) + 1

            result_type = entry.get("result", {}).get("type", "")
            if result_type == "success":
                success_count += 1
            elif result_type == "error":
                error_count += 1

        return {
            "total_count": len(entries),
            "success_count": success_count,
            "error_count": error_count,
            "grader_counts": grader_counts,
        }


# Singleton instance getter
_logger_instance: SingleEvaluationLogger | None = None


def get_single_evaluation_logger() -> SingleEvaluationLogger:
    """Get the singleton logger instance.

    Returns:
        SingleEvaluationLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SingleEvaluationLogger()
    return _logger_instance


def log_single_evaluation(
    grader_name: str,
    input_data: dict[str, Any],
    result: Any,
    threshold: float,
    elapsed_time: float,
    model_name: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to log a single evaluation.

    Args:
        grader_name: Name of the grader used
        input_data: Input data (query, response, etc.)
        result: Evaluation result object
        threshold: Pass/fail threshold used
        elapsed_time: Time taken for evaluation in seconds
        model_name: Name of the LLM model used (if any)
        extra_params: Any extra parameters used

    Returns:
        True if logging was successful
    """
    # Re-initialize logger to use current workspace
    logger_instance = SingleEvaluationLogger()
    return logger_instance.log_evaluation(
        grader_name=grader_name,
        input_data=input_data,
        result=result,
        threshold=threshold,
        elapsed_time=elapsed_time,
        model_name=model_name,
        extra_params=extra_params,
    )
