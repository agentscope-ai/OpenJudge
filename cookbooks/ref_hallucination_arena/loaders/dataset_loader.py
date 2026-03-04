# -*- coding: utf-8 -*-
"""Dataset loader: load evaluation queries from user-provided JSON/JSONL files.

Supported formats:
  - .json  : a JSON array of query objects
  - .jsonl : one query object per line

Each query object must have a "query" field. Optional fields:
  - "discipline": academic discipline category
  - "num_refs": expected number of references to recommend
  - "language": query language (zh / en)
  - "metadata": arbitrary extra info
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import QueryItem


class DatasetLoader:
    """Load and validate evaluation queries from JSON/JSONL files.

    Example:
        >>> loader = DatasetLoader("examples/queries.json")
        >>> queries = loader.load()
        >>> print(f"Loaded {len(queries)} queries")
    """

    SUPPORTED_FORMATS = {".json", ".jsonl"}

    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize loader.

        Args:
            dataset_path: Path to JSON or JSONL dataset file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
        """
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        if self.path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{self.path.suffix}'. " f"Supported: {self.SUPPORTED_FORMATS}")

    def load(self) -> List[QueryItem]:
        """Load and validate all queries from the dataset file.

        Returns:
            List of validated QueryItem objects.
        """
        raw_items = self._read_file()
        queries = self._validate(raw_items)
        logger.info(f"Loaded {len(queries)} queries from {self.path}")

        # Log discipline distribution
        disciplines = {}
        for q in queries:
            d = q.discipline or "unspecified"
            disciplines[d] = disciplines.get(d, 0) + 1
        logger.info(f"Discipline distribution: {disciplines}")

        return queries

    def _read_file(self) -> List[Dict[str, Any]]:
        """Read raw data from file."""
        if self.path.suffix == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "queries" in data:
                # Also support {"queries": [...], "metadata": {...}} wrapper format
                return data["queries"]
            else:
                raise ValueError("JSON file must be an array of query objects, " "or an object with a 'queries' key.")
        else:  # .jsonl
            items = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
            return items

    def _validate(self, raw_items: List[Dict[str, Any]]) -> List[QueryItem]:
        """Validate raw dicts into QueryItem objects."""
        queries = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                logger.warning(f"Skipping item {idx}: not a dict")
                continue
            if "query" not in item:
                logger.warning(f"Skipping item {idx}: missing 'query' field")
                continue
            try:
                queries.append(QueryItem(**item))
            except Exception as e:
                logger.warning(f"Skipping item {idx}: validation error: {e}")
        return queries
