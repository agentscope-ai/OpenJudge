# -*- coding: utf-8 -*-
"""Bounded offline JSON replay of native GraderScore/GraderError partitions."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openjudge.calibration.artifacts import MAX_JSON_BYTES, read_json
from openjudge.calibration.contracts import PartitionRole, canonical_digest
from openjudge.graders.schema import GraderError, GraderResult, GraderScore

ROLES = {"calibration", "validation", "test", "analysis", "inference"}


class CachedPartition:
    """Detached snapshot of paired native input/results, preserving dataset order.

    Cache files contain supplied samples and native metadata/reasons. Callers
    should project sensitive input fields before saving; summary workflow reports
    intentionally omit those payloads. A cache is data, never an executable class.

    Args:
        dataset: Explicit labeled rows including stable sample and task/group IDs.
        grader_results: Aligned native pointwise score/error results.
        role: Declared experimental role, enforced again by the workflow.
    """

    def __init__(
        self,
        dataset: Sequence[dict[str, Any]],
        grader_results: Sequence[GraderResult],
        *,
        role: PartitionRole,
    ) -> None:
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError("unknown calibration partition role")
        if len(dataset) != len(grader_results):
            raise ValueError("dataset and grader_results must have the same length")
        if any(not isinstance(sample, dict) for sample in dataset):
            raise ValueError("cached dataset rows must be dictionaries")
        if any(not isinstance(result, (GraderScore, GraderError)) for result in grader_results):
            raise ValueError("cached results must be native GraderScore or GraderError")
        self.dataset = tuple(copy.deepcopy(sample) for sample in dataset)
        self.grader_results = tuple(result.model_copy(deep=True) for result in grader_results)
        self.role = role

    @classmethod
    def from_runner(
        cls,
        dataset: Sequence[dict[str, Any]],
        runner_results: Mapping[str, Sequence[GraderResult]],
        grader_name: str,
        *,
        role: PartitionRole,
    ) -> "CachedPartition":
        """Snapshot one native GradingRunner result list after explicit grader selection."""
        if grader_name not in runner_results:
            raise ValueError(f"runner results do not contain grader {grader_name!r}")
        return cls(dataset, runner_results[grader_name], role=role)

    def to_payload(self) -> dict[str, Any]:
        """Return a detached primitive envelope with explicit native schema tags."""
        records = []
        for sample, result in zip(self.dataset, self.grader_results):
            records.append(
                {
                    "sample": copy.deepcopy(sample),
                    "result_type": "error" if isinstance(result, GraderError) else "score",
                    "result": result.model_dump(mode="json"),
                }
            )
        envelope = {"version": 1, "kind": "native_calibration_partition", "role": self.role, "records": records}
        return {**envelope, "content_id": canonical_digest(envelope)}

    def save(self, path: str | Path) -> None:
        """Save paired inputs/results as finite JSON; no pickle or class path is stored."""
        payload = self.to_payload()
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> "CachedPartition":
        """Read bounded JSON and reject unknown cache/native schema fields early."""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        with Path(path).open("rb") as stream:
            payload = read_json(stream.read(max_bytes + 1), max_bytes)
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Any) -> "CachedPartition":
        """Validate the finite primitive envelope before constructing native objects."""
        keys = {"version", "kind", "role", "records", "content_id"}
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ValueError("invalid cache envelope fields")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("unsupported cache version")
        if (
            payload["kind"] != "native_calibration_partition"
            or not isinstance(payload["role"], str)
            or payload["role"] not in ROLES
        ):
            raise ValueError("unsupported cache kind or partition role")
        if not isinstance(payload["records"], list):
            raise ValueError("cache records must be a list")
        expected = canonical_digest({key: value for key, value in payload.items() if key != "content_id"})
        if payload["content_id"] != expected:
            raise ValueError("cache content checksum mismatch")
        dataset = []
        results = []
        for index, record in enumerate(payload["records"]):
            if not isinstance(record, dict) or set(record) != {"sample", "result_type", "result"}:
                raise ValueError(f"invalid cache record at index {index}")
            if not isinstance(record["sample"], dict) or not isinstance(record["result"], dict):
                raise ValueError(f"invalid native sample/result at index {index}")
            if not isinstance(record["result_type"], str):
                raise ValueError(f"unsupported native result type at index {index}")
            result_class = {"score": GraderScore, "error": GraderError}.get(record["result_type"])
            if result_class is None or set(record["result"]) - set(result_class.model_fields):
                raise ValueError(f"unsupported native result schema at index {index}")
            result = result_class.model_validate(record["result"])
            dataset.append(record["sample"])
            results.append(result)
        return cls(dataset, results, role=payload["role"])
