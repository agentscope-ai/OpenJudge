# -*- coding: utf-8 -*-
"""Checkpoint management for evaluation pipeline."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from cookbooks.zero_shot_evaluation.schema import GeneratedQuery


class EvaluationStage(str, Enum):
    """Evaluation pipeline stages."""

    NOT_STARTED = "not_started"
    QUERIES_GENERATED = "queries_generated"
    RESPONSES_COLLECTED = "responses_collected"
    RUBRICS_GENERATED = "rubrics_generated"
    EVALUATION_COMPLETE = "evaluation_complete"


class CheckpointData(BaseModel):
    """Checkpoint data model."""

    stage: EvaluationStage = Field(default=EvaluationStage.NOT_STARTED)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Data files
    queries_file: Optional[str] = None
    responses_file: Optional[str] = None
    rubrics_file: Optional[str] = None

    # Progress tracking
    total_queries: int = 0
    collected_responses: int = 0
    evaluated_pairs: int = 0
    total_pairs: int = 0


class CheckpointManager:
    """Manage evaluation checkpoints for resume capability."""

    CHECKPOINT_FILE = "checkpoint.json"
    QUERIES_FILE = "queries.json"
    RESPONSES_FILE = "responses.json"
    RUBRICS_FILE = "rubrics.json"

    def __init__(self, output_dir: str):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to store checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint: Optional[CheckpointData] = None

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.CHECKPOINT_FILE

    def load(self) -> Optional[CheckpointData]:
        """Load existing checkpoint if available."""
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._checkpoint = CheckpointData(**data)
            logger.info(f"Loaded checkpoint: stage={self._checkpoint.stage.value}")
            return self._checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint to file."""
        checkpoint.updated_at = datetime.now().isoformat()
        self._checkpoint = checkpoint

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.model_dump(), f, indent=2, ensure_ascii=False)

        logger.debug(f"Checkpoint saved: stage={checkpoint.stage.value}")

    def save_queries(self, queries: List[GeneratedQuery]) -> str:
        """Save generated queries."""
        file_path = self.output_dir / self.QUERIES_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([q.model_dump() for q in queries], f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(queries)} queries to {file_path}")
        return str(file_path)

    def load_queries(self) -> List[GeneratedQuery]:
        """Load saved queries."""
        file_path = self.output_dir / self.QUERIES_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = [GeneratedQuery(**item) for item in data]
        logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries

    def save_responses(self, responses: List[Dict[str, Any]]) -> str:
        """Save collected responses."""
        file_path = self.output_dir / self.RESPONSES_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(responses)} responses to {file_path}")
        return str(file_path)

    def load_responses(self) -> List[Dict[str, Any]]:
        """Load saved responses."""
        file_path = self.output_dir / self.RESPONSES_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            responses = json.load(f)

        logger.info(f"Loaded {len(responses)} responses from {file_path}")
        return responses

    def save_rubrics(self, rubrics: List[str]) -> str:
        """Save generated rubrics."""
        file_path = self.output_dir / self.RUBRICS_FILE

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(rubrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(rubrics)} rubrics to {file_path}")
        return str(file_path)

    def load_rubrics(self) -> List[str]:
        """Load saved rubrics."""
        file_path = self.output_dir / self.RUBRICS_FILE

        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            rubrics = json.load(f)

        logger.info(f"Loaded {len(rubrics)} rubrics from {file_path}")
        return rubrics

    def update_stage(
        self,
        stage: EvaluationStage,
        **kwargs,
    ) -> None:
        """Update checkpoint stage and save."""
        if self._checkpoint is None:
            self._checkpoint = CheckpointData()

        self._checkpoint.stage = stage
        for key, value in kwargs.items():
            if hasattr(self._checkpoint, key):
                setattr(self._checkpoint, key, value)

        self.save(self._checkpoint)

    def clear(self) -> None:
        """Clear all checkpoint data."""
        for file_name in [
            self.CHECKPOINT_FILE,
            self.QUERIES_FILE,
            self.RESPONSES_FILE,
            self.RUBRICS_FILE,
        ]:
            file_path = self.output_dir / file_name
            if file_path.exists():
                file_path.unlink()

        self._checkpoint = None
        logger.info("Checkpoint cleared")
