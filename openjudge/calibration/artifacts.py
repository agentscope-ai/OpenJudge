# -*- coding: utf-8 -*-
"""Versioned, checksummed primitive JSON artifacts; no executable deserialization."""

import json
import math
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import Field, field_validator, model_validator

from openjudge.calibration.contracts import (
    CalibrationContract,
    CalibrationScope,
    CompatibilityOverride,
    FrozenModel,
    canonical_digest,
    probability,
)
from openjudge.calibration.splits import PartitionProvenance, assert_disjoint

MAX_JSON_BYTES = 16 * 1024 * 1024
ArtifactType = TypeVar("ArtifactType", bound="JSONArtifact")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently keeping the last."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Python's nonstandard NaN/Infinity tokens are not artifact JSON."""
    raise ValueError(f"nonfinite JSON value: {value}")


def read_json(payload: str | bytes, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Parse bounded finite JSON, rejecting ambiguous keys and malformed text."""
    size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
    if size > max_bytes:
        raise ValueError(f"JSON exceeds {max_bytes} bytes")
    try:
        return json.loads(payload, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("invalid artifact JSON") from exc


class JSONArtifact(FrozenModel):
    """Base for inspectable content-addressed JSON; checksum is not authentication."""

    version: Literal[1] = 1
    artifact_id: str

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> Any:
        """Do not let True masquerade as schema version one."""
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value

    @model_validator(mode="after")
    def checksum(self) -> "JSONArtifact":
        """Validate all declared content against its reproducible ID."""
        payload = self.model_dump(mode="json", exclude={"artifact_id"})
        if self.artifact_id != canonical_digest(payload):
            raise ValueError("artifact content checksum mismatch")
        return self

    @classmethod
    def create(cls: type[ArtifactType], **values: Any) -> ArtifactType:
        """Build a checked artifact after normalizing nested model defaults."""
        # Construct only for hashing, then run full validation before returning.
        candidate = cls.model_construct(version=1, artifact_id="", **values)
        normalized = candidate.model_dump(mode="json", exclude={"artifact_id"})
        return cls.model_validate({**normalized, "artifact_id": canonical_digest(normalized)})

    @classmethod
    def model_validate_json(cls: type[ArtifactType], json_data: str | bytes | bytearray, **kwargs: Any) -> ArtifactType:
        """Apply strict JSON parsing in addition to Pydantic field validation."""
        return cls.model_validate(
            read_json(bytes(json_data) if isinstance(json_data, bytearray) else json_data), **kwargs
        )

    def save(self, path: str | Path) -> None:
        """Write finite readable JSON; callers choose the local destination."""
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls: type[ArtifactType], path: str | Path) -> ArtifactType:
        """Read at most the artifact limit before validating any state."""
        with Path(path).open("rb") as stream:
            return cls.model_validate_json(stream.read(MAX_JSON_BYTES + 1))


class FitSettings(FrozenModel):
    """Settings for sigmoid-on-raw-confidence or isotonic postprocessing."""

    method: Literal["identity", "sigmoid", "isotonic"] = "identity"
    regularization: float = Field(default=1.0, gt=0, description="Inverse L2 regularization C for sklearn sigmoid")
    max_iterations: int = Field(default=1000, ge=1, strict=True)
    tolerance: float = Field(default=1e-9, gt=0)


class FitState(FrozenModel):
    """Only the primitive numerical state needed for deterministic inference."""

    kind: Literal["identity", "constant", "sigmoid", "isotonic"]
    constant: float | None = None
    slope: float | None = None
    intercept: float | None = None
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()

    @field_validator("constant", mode="before")
    @classmethod
    def rate(cls, value: Any) -> float | None:
        """Constant models represent a valid empirical correctness rate."""
        return probability(value) if value is not None else None

    @field_validator("x_thresholds", "y_thresholds", mode="before")
    @classmethod
    def knots(cls, values: Any) -> tuple[float, ...]:
        """Knots are bounded probabilities, not executable estimator objects."""
        return tuple(probability(value) for value in values)

    @model_validator(mode="after")
    def coherent_state(self) -> "FitState":
        """Reject surplus, absent or nonmonotone estimator state."""
        if self.kind == "identity":
            valid = self.constant is None and self.slope is None and self.intercept is None and not self.x_thresholds
        elif self.kind == "constant":
            valid = (
                self.constant is not None and self.slope is None and self.intercept is None and not self.x_thresholds
            )
        elif self.kind == "sigmoid":
            valid = self.constant is None and self.slope is not None and self.slope >= 0
            valid = valid and self.intercept is not None and not self.x_thresholds
        else:
            valid = self.constant is None and self.slope is None and self.intercept is None and bool(self.x_thresholds)
            valid = valid and len(self.x_thresholds) == len(self.y_thresholds)
            valid = valid and all(a < b for a, b in zip(self.x_thresholds, self.x_thresholds[1:]))
            valid = valid and all(a <= b for a, b in zip(self.y_thresholds, self.y_thresholds[1:]))
        if self.kind != "isotonic" and self.y_thresholds:
            valid = False
        if not valid:
            raise ValueError("incoherent or nonmonotone calibrator state")
        return self


class FitArtifact(JSONArtifact):
    """Fitted event, scope, settings, data provenance and safe inference state."""

    kind: Literal["calibrator"] = "calibrator"
    contract: CalibrationContract
    scope: CalibrationScope
    settings: FitSettings
    state: FitState
    source: PartitionProvenance
    correct_count: int = Field(ge=0)
    confidence_range: tuple[float, float]
    domain_counts: dict[str, int]
    degeneracy: str | None = None
    implementation: str = "openjudge.correctness-calibration.v1"

    @model_validator(mode="after")
    def fit_contract(self) -> "FitArtifact":
        """A fitted state must have actual calibration-role support."""
        if self.source.role != "calibration" or not 0 <= self.correct_count <= self.source.valid_count:
            raise ValueError("invalid calibration source or outcome support")
        if not self.source.valid_count:
            raise ValueError("fitting requires valid calibration observations")
        if not 0 <= self.confidence_range[0] <= self.confidence_range[1] <= 1:
            raise ValueError("invalid fitted confidence range")
        allowed = {self.settings.method, "constant"} if self.settings.method != "identity" else {"identity"}
        if self.state.kind not in allowed:
            raise ValueError("state does not match calibration method")
        if any(count < 0 for count in self.domain_counts.values()):
            raise ValueError("domain counts cannot be negative")
        return self


class SelectionEvidence(FrozenModel):
    """Validation operating point retained without exposing row-level labels."""

    accepted_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    risk: float | None
    coverage: float

    @model_validator(mode="after")
    def coherent_counts(self) -> "SelectionEvidence":
        """An empty accepted set cannot claim zero risk."""
        if self.error_count > self.accepted_count:
            raise ValueError("more accepted errors than accepted observations")
        expected = self.error_count / self.accepted_count if self.accepted_count else None
        if self.risk != expected or not 0 <= self.coverage <= 1:
            raise ValueError("inconsistent validation selection evidence")
        return self


def hoeffding_guarantee(alpha: float) -> str:
    """Describe exactly the assumptions used by the optional simultaneous bound."""
    return (
        "fixed-grid Bonferroni-Hoeffding upper risk bounds at alpha="
        + str(alpha)
        + "; conditional on independent identically distributed observation pairs, one observation per group, "
        "an independently fitted calibrator and the grid declared before validation inspection; "
        "not guaranteed under distribution shift"
    )


class PolicyArtifact(JSONArtifact):
    """A threshold chosen once on validation data, bound to a fitted artifact."""

    kind: Literal["selective_policy"] = "selective_policy"
    fit_artifact_id: str
    contract: CalibrationContract
    scope: CalibrationScope
    fit_source: PartitionProvenance
    validation_source: PartitionProvenance
    threshold: float | None
    risk_target: float
    min_accepted: int = Field(ge=1)
    criterion: Literal["empirical", "hoeffding"]
    candidates: tuple[float, ...]
    candidate_source: Literal["validation_ties", "fixed_grid"]
    candidate_count: int = Field(ge=0)
    alpha: float
    assume_iid_observations: bool
    selection: SelectionEvidence
    selection_upper_risk: float | None
    reason: Literal["feasible_threshold", "no_feasible_threshold"]
    guarantee: str | None
    overrides: tuple[CompatibilityOverride, ...] = ()

    @field_validator("threshold", "selection_upper_risk", mode="before")
    @classmethod
    def optional_probability(cls, value: Any) -> float | None:
        """No infinite sentinel is needed for abstain-all."""
        return probability(value) if value is not None else None

    @field_validator("risk_target", "alpha", mode="before")
    @classmethod
    def bounded_probability(cls, value: Any) -> float:
        """Serialized statistical targets are explicit probabilities."""
        return probability(value)

    @model_validator(mode="after")
    def selection_contract(self) -> "PolicyArtifact":
        """Reject misleading guarantees, incompatible sources and fake empty success."""
        if self.fit_source.role != "calibration" or self.validation_source.role != "validation":
            raise ValueError("policy requires calibration and validation sources")
        assert_disjoint(self.fit_source, self.validation_source)
        if self.candidate_count != len(self.candidates) or tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError("invalid threshold candidate grid")
        for candidate in self.candidates:
            probability(candidate)
        if self.threshold is None:
            if self.selection.accepted_count or self.reason != "no_feasible_threshold":
                raise ValueError("abstain-all policy has inconsistent support")
        elif self.threshold not in self.candidates or self.reason != "feasible_threshold":
            raise ValueError("selected threshold was not a declared candidate")
        elif self.selection.accepted_count < self.min_accepted:
            raise ValueError("selected threshold lacks required validation support")
        count = self.selection.accepted_count
        if count > self.validation_source.valid_count:
            raise ValueError("policy accepts more than the validation support")
        expected_coverage = count / self.validation_source.sample_count if self.validation_source.sample_count else 0
        if self.selection.coverage != expected_coverage:
            raise ValueError("policy coverage does not match its validation denominator")
        expected_upper = self.selection.risk
        if self.criterion == "empirical":
            if self.guarantee is not None:
                raise ValueError("empirical threshold selection carries no population guarantee")
        else:
            if not self.assume_iid_observations or not 0 < self.alpha < 1 or self.candidate_source != "fixed_grid":
                raise ValueError("bound requires fixed-grid and explicit independent observation assumptions")
            if not self.candidate_count or self.guarantee != hoeffding_guarantee(self.alpha):
                raise ValueError("bound assumptions and interpretation must be preserved exactly")
            if count:
                expected_upper = min(
                    1.0, self.selection.risk + math.sqrt(math.log(self.candidate_count / self.alpha) / (2 * count))
                )
        if self.selection_upper_risk != expected_upper:
            raise ValueError("policy upper risk is inconsistent with its selection criterion")
        if self.threshold is not None and expected_upper > self.risk_target:
            raise ValueError("selected policy does not meet its requested empirical criterion")
        return self


def check_compatibility(
    contract: CalibrationContract,
    scope: CalibrationScope,
    fitted_contract: CalibrationContract,
    fitted_scope: CalibrationScope,
    override_reason: str | None = None,
) -> CompatibilityOverride | None:
    """Fail closed on event/extractor/scope differences, or record an explicit override."""
    mismatches = []
    if contract != fitted_contract:
        mismatches.append("contract")
    if scope != fitted_scope:
        mismatches.append("scope")
    if mismatches:
        if override_reason is None:
            raise ValueError("incompatible calibration " + ", ".join(mismatches))
        return CompatibilityOverride(mismatches=tuple(mismatches), reason=override_reason)
    return None
