# -*- coding: utf-8 -*-
"""Data parser service for Auto Rubric feature.

Parses and validates uploaded data files for Iterative Rubric generation.
Supports JSON, JSONL, and CSV formats.
"""

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ParseResult:
    """Result of data parsing.

    Attributes:
        success: Whether parsing was successful.
        data: List of parsed data dictionaries.
        total_count: Total number of records.
        error: Error message if parsing failed.
        warnings: List of warning messages.
        min_score: Minimum score found in data (for pointwise mode).
        max_score: Maximum score found in data (for pointwise mode).
    """

    success: bool
    data: list[dict[str, Any]] | None = None
    total_count: int = 0
    error: str | None = None
    warnings: list[str] | None = None
    min_score: int | None = None
    max_score: int | None = None


class DataParser:
    """Parser for evaluation data files.

    Supports:
    - JSON: {"data": [...]} or [...]
    - JSONL: One JSON object per line
    - CSV: Header row with column names

    Expected fields for Pointwise mode:
    - query: The input query (required)
    - response: The response to evaluate (required)
    - label_score: The expected score (required for training)

    Expected fields for Pairwise/Listwise mode:
    - query: The input query (required)
    - responses: List of responses to rank (required)
    - label_rank: Expected ranking (required for training)
    """

    # Maximum records allowed (no hard limit, user can select sample size in UI)
    MAX_RECORDS = 100000

    # Required fields for different modes
    POINTWISE_REQUIRED = {"query", "response", "label_score"}
    LISTWISE_REQUIRED = {"query", "responses", "label_rank"}

    def parse_file(
        self,
        file_content: bytes,
        filename: str,
        mode: str = "pointwise",
    ) -> ParseResult:
        """Parse an uploaded file.

        Args:
            file_content: Raw file content bytes.
            filename: Original filename (used to determine format).
            mode: Evaluation mode ("pointwise" or "listwise").

        Returns:
            ParseResult with parsed data or error information.
        """
        try:
            # Determine file format
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

            if ext == "json":
                return self._parse_json(file_content, mode)
            elif ext == "jsonl":
                return self._parse_jsonl(file_content, mode)
            elif ext == "csv":
                return self._parse_csv(file_content, mode)
            else:
                return ParseResult(
                    success=False,
                    error=f"Unsupported file format: .{ext}. Supported: .json, .jsonl, .csv",
                )

        except Exception as e:
            logger.error(f"Failed to parse file {filename}: {e}")
            return ParseResult(
                success=False,
                error=f"Failed to parse file: {str(e)}",
            )

    def _parse_json(self, content: bytes, mode: str) -> ParseResult:
        """Parse JSON file content."""
        try:
            text = content.decode("utf-8")
            data = json.loads(text)

            # Handle {"data": [...]} format
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if not isinstance(data, list):
                return ParseResult(
                    success=False,
                    error='JSON must be an array or {"data": [...]} format',
                )

            return self._validate_and_return(data, mode)

        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                error=f"Invalid JSON format: {str(e)}",
            )

    def _parse_jsonl(self, content: bytes, mode: str) -> ParseResult:
        """Parse JSONL file content (one JSON object per line)."""
        try:
            text = content.decode("utf-8")
            lines = text.strip().split("\n")
            data = []

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    return ParseResult(
                        success=False,
                        error=f"Invalid JSON at line {i}: {str(e)}",
                    )

            return self._validate_and_return(data, mode)

        except Exception as e:
            return ParseResult(
                success=False,
                error=f"Failed to parse JSONL: {str(e)}",
            )

    def _parse_csv(self, content: bytes, mode: str) -> ParseResult:
        """Parse CSV file content."""
        try:
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            data = []

            for row in reader:
                # Convert empty strings to None
                item = {k: (v if v else None) for k, v in row.items()}

                # Try to parse label_score as number
                if "label_score" in item and item["label_score"]:
                    try:
                        item["label_score"] = float(item["label_score"])
                    except ValueError:
                        pass

                # Try to parse label_rank as list
                if "label_rank" in item and item["label_rank"]:
                    try:
                        item["label_rank"] = json.loads(item["label_rank"])
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Try to parse responses as list (for listwise mode)
                if "responses" in item and item["responses"]:
                    try:
                        item["responses"] = json.loads(item["responses"])
                    except (json.JSONDecodeError, ValueError):
                        pass

                data.append(item)

            return self._validate_and_return(data, mode)

        except Exception as e:
            return ParseResult(
                success=False,
                error=f"Failed to parse CSV: {str(e)}",
            )

    def _validate_and_return(
        self,
        data: list[dict[str, Any]],
        mode: str,
    ) -> ParseResult:
        """Validate parsed data and return result."""
        if not data:
            return ParseResult(
                success=False,
                error="No data found in file",
            )

        if len(data) > self.MAX_RECORDS:
            return ParseResult(
                success=False,
                error=f"Too many records: {len(data)}. Maximum allowed: {self.MAX_RECORDS}",
            )

        # Determine required fields based on mode
        required = self.POINTWISE_REQUIRED if mode == "pointwise" else self.LISTWISE_REQUIRED

        # Validate each record
        warnings = []
        valid_data = []

        for i, item in enumerate(data, 1):
            item_keys = set(item.keys())
            missing = required - item_keys

            if missing:
                warnings.append(f"Record {i}: missing fields {missing}")
                continue

            # Check for None/empty required values
            has_empty = False
            for field in required:
                if item.get(field) is None or item.get(field) == "":
                    has_empty = True
                    warnings.append(f"Record {i}: empty value for '{field}'")
                    break

            if not has_empty:
                valid_data.append(item)

        if not valid_data:
            return ParseResult(
                success=False,
                error="No valid records found. Check that all required fields are present.",
                warnings=warnings if warnings else None,
            )

        # Extract score range for pointwise mode
        min_score = None
        max_score = None
        if mode == "pointwise":
            scores = []
            for item in valid_data:
                score = item.get("label_score")
                if score is not None:
                    try:
                        scores.append(int(float(score)))
                    except (ValueError, TypeError):
                        pass
            if scores:
                min_score = min(scores)
                max_score = max(scores)

        return ParseResult(
            success=True,
            data=valid_data,
            total_count=len(valid_data),
            warnings=warnings if warnings else None,
            min_score=min_score,
            max_score=max_score,
        )

    def get_preview(
        self,
        data: list[dict[str, Any]],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Get a preview of the parsed data.

        Args:
            data: Parsed data list.
            max_items: Maximum number of items to preview.

        Returns:
            List of preview items with truncated content.
        """
        preview = []
        for item in data[:max_items]:
            preview_item = {}
            for k, v in item.items():
                if isinstance(v, str) and len(v) > 100:
                    preview_item[k] = v[:100] + "..."
                elif isinstance(v, list) and len(v) > 3:
                    preview_item[k] = v[:3] + ["..."]
                else:
                    preview_item[k] = v
            preview.append(preview_item)
        return preview
