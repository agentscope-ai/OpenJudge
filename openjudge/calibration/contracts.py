# -*- coding: utf-8 -*-
"""Immutable contracts for the probability that a judge decision is correct."""

import hashlib
import json
import math
from collections import Counter
from numbers import Real
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

PartitionRole = Literal["calibration", "validation", "test", "analysis", "inference"]
ObservationStatus = Literal[
    "valid", "provider_error", "user_excluded", "missing_label", "invalid_label",
    "missing_prediction", "invalid_prediction", "missing_confidence", "invalid_confidence", "unsupported_result",
]
DecisionLabel = str | int | float


class FrozenModel(BaseModel):
    """Reject unknown fields and mutation at the numerical/artifact boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def canonical_digest(value: Any) -> str:
    """Hash finite primitive JSON deterministically, independent of object key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def probability(value: Any) -> float:
    """Validate a probability without coercing strings, booleans or invalid ranges."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("confidence must be a real number, not a string or boolean")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("confidence must be finite and in [0, 1]")
    return number


def valid_identifier(value: Any) -> str:
    """Require a stable nonempty string; do not normalize identities silently."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identity must be a nonempty string")
    return value


def is_decision_label(value: Any) -> bool:
    """Accept finite scalar class labels; booleans cannot alias numeric classes."""
    return not isinstance(value, bool) and (
        isinstance(value, str) or isinstance(value, Real) and math.isfinite(float(value))
    )


class CalibrationContract(FrozenModel):
    """Declare an independently labeled binary correctness event and its inputs.

    Args:
        target_id: Versioned meaning of the correctness event.
        confidence_path: Explicit result/dataset path, or None for a named callback.
        decision_labels: Two allowed decisions; correctness is equality with label.

    Example:
        >>> CalibrationContract(target_id="matches_reference_v1", confidence_path="metadata.confidence").label_path
        'label'
    """

    target_id: str
    confidence_path: str | None
    confidence_source: Literal["result", "dataset"] = "result"
    confidence_interpretation: str = "probability that predicted decision equals independently supplied label"
    extractor_id: str | None = None
    prediction_path: str = "score"
    label_path: str = "label"
    sample_id_path: str = "id"
    group_id_path: str = "group"
    domain_path: str | None = None
    exclude_path: str | None = None
    decision_labels: tuple[DecisionLabel, DecisionLabel] = (0, 1)

    @field_validator("target_id", "confidence_interpretation", "prediction_path", "label_path",
                     "sample_id_path", "group_id_path")
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Reject missing event and field identities."""
        return valid_identifier(value)

    @field_validator("confidence_path", "extractor_id", "domain_path", "exclude_path")
    @classmethod
    def optional_path(cls, value: str | None) -> str | None:
        """Optional paths are either absent or nonempty."""
        return valid_identifier(value) if value is not None else None

    @field_validator("decision_labels", mode="before")
    @classmethod
    def labels(cls, value: Any) -> Any:
        """Reject ambiguous or unsupported decision spaces."""
        if len(value) != 2 or not all(is_decision_label(v) for v in value) or value[0] == value[1]:
            raise ValueError("decision_labels must contain two distinct finite scalar labels")
        return value

    @model_validator(mode="after")
    def confidence_contract(self) -> "CalibrationContract":
        """Require one reproducible confidence extraction mechanism."""
        if (self.confidence_path is None) == (self.extractor_id is None):
            raise ValueError("declare exactly one confidence_path or extractor_id")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the full event/extraction contract identity."""
        return canonical_digest(self.model_dump(mode="json"))


class CalibrationScope(FrozenModel):
    """Identify the judge, model/prompt version and application domain explicitly."""

    judge_id: str
    model_id: str
    scope_id: str

    @field_validator("judge_id", "model_id", "scope_id")
    @classmethod
    def identity(cls, value: str) -> str:
        """Require versioned, nonempty identities supplied by the caller."""
        return valid_identifier(value)


class CompatibilityOverride(FrozenModel):
    """Record a caller's explicit decision to apply outside fitted compatibility."""

    mismatches: tuple[str, ...]
    reason: str

    @field_validator("reason")
    @classmethod
    def explanation(cls, value: str) -> str:
        """An override must be reviewable, not an empty flag."""
        return valid_identifier(value)


class Observation(FrozenModel):
    """Detached row; intentionally contains no prompts, native result or target label."""

    sample_id: str
    group_id: str
    status: ObservationStatus
    probability: float | None = None
    correctness: Literal[0, 1] | None = None
    domain: str | None = None

    @field_validator("sample_id", "group_id")
    @classmethod
    def identity(cls, value: str) -> str:
        """Rows always have stable identities, including excluded rows."""
        return valid_identifier(value)

    @field_validator("probability", mode="before")
    @classmethod
    def confidence(cls, value: Any) -> float | None:
        """Validate explicitly, including bool rejection."""
        return probability(value) if value is not None else None

    @model_validator(mode="after")
    def consistent_status(self) -> "Observation":
        """Only usable rows may carry metric inputs."""
        if self.status == "valid" and (self.probability is None or self.correctness is None):
            raise ValueError("valid observations require probability and correctness")
        if self.status != "valid" and (self.probability is not None or self.correctness is not None):
            raise ValueError("excluded observations cannot carry metric inputs")
        return self


class ObservationBatch(FrozenModel):
    """Ordered immutable observations with explicit partition, event and scope."""

    observations: tuple[Observation, ...]
    contract: CalibrationContract
    scope: CalibrationScope
    role: PartitionRole = "analysis"
    overrides: tuple[CompatibilityOverride, ...] = ()
    fit_artifact_id: str | None = None

    @model_validator(mode="after")
    def unique_samples(self) -> "ObservationBatch":
        """Duplicate trajectories require distinct sample IDs in one group."""
        ids = [row.sample_id for row in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample IDs")
        return self

    @property
    def valid(self) -> tuple[Observation, ...]:
        """Return numerical rows without discarding the original denominator."""
        return tuple(row for row in self.observations if row.status == "valid")

    @property
    def counts(self) -> dict[str, int]:
        """Return a detached status counter; zero counts remain unambiguous."""
        return dict(Counter(row.status for row in self.observations))
