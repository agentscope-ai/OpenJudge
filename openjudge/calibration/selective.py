# -*- coding: utf-8 -*-
"""Tie-aware selective risk and validation-only acceptance policy selection."""

import bisect
import math
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from openjudge.calibration.artifacts import FitArtifact, PolicyArtifact, SelectionEvidence
from openjudge.calibration.contracts import FrozenModel, ObservationBatch, probability
from openjudge.calibration.fitting import apply_calibrator
from openjudge.calibration.splits import assert_disjoint, partition_provenance


class SelectionSettings(FrozenModel):
    """Empirical selection or an explicitly assumed fixed-grid Hoeffding bound."""

    risk_target: float = Field(default=0.1, ge=0, le=1)
    min_accepted: int = Field(default=1, ge=1, strict=True)
    criterion: Literal["empirical", "hoeffding"] = "empirical"
    candidates: tuple[float, ...] | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)
    assume_iid_observations: bool = False

    @field_validator("candidates", mode="before")
    @classmethod
    def candidate_grid(cls, values: Sequence[float] | None) -> tuple[float, ...] | None:
        """Preserve only a canonical, finite declared grid."""
        if values is None:
            return None
        candidates = tuple(probability(value) for value in values)
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate grid must be unique and increasing")
        return candidates

    @field_validator("risk_target", "alpha", mode="before")
    @classmethod
    def statistical_probability(cls, value: Any) -> float:
        """Boolean/string settings cannot imply a statistical confidence level."""
        return probability(value)

    @model_validator(mode="after")
    def valid_bound(self) -> "SelectionSettings":
        """Data-derived tie thresholds carry descriptive empirical selection only."""
        if self.criterion == "hoeffding":
            if not self.candidates:
                raise ValueError("hoeffding selection requires a nonempty fixed candidate grid")
            if not self.assume_iid_observations:
                raise ValueError("hoeffding selection requires explicitly acknowledged independent observations")
        return self


class SelectiveMetrics(FrozenModel):
    """Operating counts over all inputs and risk over accepted valid decisions."""

    threshold: float | None
    total_count: int
    valid_count: int
    accepted_count: int
    correct_count: int
    error_count: int
    abstained_count: int
    provider_error_count: int
    excluded_count: int
    risk: float | None
    accepted_accuracy: float | None
    coverage: float
    evaluable_coverage: float
    unconditional_error_count: int
    unconditional_error_rate: float | None
    known_wrong_count: int
    error_definition: str = "accepted incorrect decisions plus provider failures, divided by all input rows"


class RiskCoveragePoint(FrozenModel):
    """Complete tie-group operating point; no arbitrary within-tie ordering."""

    threshold: float | None
    accepted_count: int
    error_count: int
    risk: float | None
    coverage: float
    evaluable_coverage: float


class RiskCoverageCurve(FrozenModel):
    """Right-endpoint step area over evaluable coverage, not failure-free total input."""

    points: tuple[RiskCoveragePoint, ...]
    aurc: float | None
    total_count: int
    valid_count: int
    interpretation: str = (
        "AURC is sum of complete tie-group width times post-group risk, normalized over valid observations; "
        "total-input coverage is separately reported and provider failures are never abstentions"
    )


def selective_metrics(batch: ObservationBatch, threshold: float | None) -> SelectiveMetrics:
    """Evaluate one fixed threshold without threshold fitting or optimistic empty risk.

    Args:
        batch: Confidence/correctness observations, retaining failed and omitted rows.
        threshold: Accept at or above this value; None intentionally abstains on all usable rows.

    Returns:
        Accepted risk/accuracy or None when empty, plus all-input coverage/failure counts.
    """
    if threshold is not None:
        threshold = probability(threshold)
    valid = batch.valid
    accepted = [row for row in valid if threshold is not None and row.probability >= threshold]
    errors = sum(1 - row.correctness for row in accepted)
    total = len(batch.observations)
    failures = batch.counts.get("provider_error", 0)
    return SelectiveMetrics(
        threshold=threshold, total_count=total, valid_count=len(valid), accepted_count=len(accepted),
        correct_count=len(accepted) - errors, error_count=errors, abstained_count=len(valid) - len(accepted),
        provider_error_count=failures, excluded_count=total - len(valid) - failures,
        risk=errors / len(accepted) if accepted else None,
        accepted_accuracy=1 - errors / len(accepted) if accepted else None,
        coverage=len(accepted) / total if total else 0,
        evaluable_coverage=len(accepted) / len(valid) if valid else 0,
        unconditional_error_count=errors + failures,
        unconditional_error_rate=(errors + failures) / total if total else None,
        known_wrong_count=sum(1 - row.correctness for row in valid),
    )


def risk_coverage_curve(batch: ObservationBatch) -> RiskCoverageCurve:
    """Add complete descending confidence tie groups and integrate their endpoint risks."""
    rows = sorted(batch.valid, key=lambda row: row.probability, reverse=True)
    total = len(batch.observations)
    count = len(rows)
    points = [RiskCoveragePoint(
        threshold=None, accepted_count=0, error_count=0, risk=None, coverage=0, evaluable_coverage=0,
    )]
    accepted = 0
    errors = 0
    area = 0.0
    while accepted < count:
        start = accepted
        threshold = rows[start].probability
        while accepted < count and rows[accepted].probability == threshold:
            errors += 1 - rows[accepted].correctness
            accepted += 1
        risk = errors / accepted
        area += (accepted - start) / count * risk
        points.append(RiskCoveragePoint(
            threshold=threshold, accepted_count=accepted, error_count=errors, risk=risk,
            coverage=accepted / total, evaluable_coverage=accepted / count,
        ))
    return RiskCoverageCurve(points=tuple(points), aurc=area if count else None, total_count=total, valid_count=count)


def _candidate_counts(batch: ObservationBatch, candidates: tuple[float, ...]) -> list[tuple[float, int, int]]:
    """Compute grid counts in O(n log n + m log n), avoiding repeated full scans."""
    rows = sorted(batch.valid, key=lambda row: row.probability)
    probabilities = [row.probability for row in rows]
    prefix_errors = [0]
    for row in rows:
        prefix_errors.append(prefix_errors[-1] + 1 - row.correctness)
    values = []
    for threshold in candidates:
        index = bisect.bisect_left(probabilities, threshold)
        values.append((threshold, len(rows) - index, prefix_errors[-1] - prefix_errors[index]))
    return values


def select_policy(
    validation: ObservationBatch, artifact: FitArtifact, settings: SelectionSettings = SelectionSettings(), *,
    override_reason: str | None = None,
) -> PolicyArtifact:
    """Select maximum validation coverage subject to the declared empirical criterion.

    Args:
        validation: Raw validation-role observations, disjoint from calibration.
        artifact: Independently fitted confidence postprocessor.
        settings: Candidate grid/tie groups, support, risk target and uncertainty criterion.
        override_reason: Recorded compatibility override for intentional domain transfer.

    Returns:
        Fixed threshold policy, or an explicit abstain-all policy if none is feasible.
        Hoeffding uses a union bound across every fixed declared threshold and
        requires one valid observation per group plus caller-acknowledged i.i.d.
        observations. These assumptions are not established by group IDs alone.
    """
    if validation.role != "validation":
        raise ValueError("select_policy requires a validation partition")
    provenance = partition_provenance(validation)
    assert_disjoint(artifact.source, provenance)
    calibrated = apply_calibrator(artifact, validation, override_reason=override_reason)
    if settings.criterion == "hoeffding":
        groups = [row.group_id for row in calibrated.valid]
        if len(groups) != len(set(groups)):
            raise ValueError("hoeffding bound requires one valid observation per independent group")
    candidates = settings.candidates
    if candidates is None:
        candidates = tuple(sorted({row.probability for row in calibrated.valid}))
    selected = None
    upper_risk = None
    for threshold, accepted, errors in _candidate_counts(calibrated, candidates):
        if accepted < settings.min_accepted:
            continue
        risk = errors / accepted
        upper = risk
        if settings.criterion == "hoeffding":
            upper = min(1.0, risk + math.sqrt(math.log(len(candidates) / settings.alpha) / (2 * accepted)))
        if upper <= settings.risk_target:
            # Increasing thresholds cannot increase support; first feasible grid
            # value maximizes support with deterministic lowest-threshold ties.
            selected = threshold
            upper_risk = upper
            break
    operating = selective_metrics(calibrated, selected)
    guarantee = None
    if settings.criterion == "hoeffding":
        guarantee = (
            "fixed-grid Bonferroni-Hoeffding upper risk bounds at alpha=" + str(settings.alpha)
            + "; conditional on independent identically distributed observation pairs, one observation per group, "
            "an independently fitted calibrator and the grid declared before validation inspection; "
            "not guaranteed under distribution shift"
        )
    return PolicyArtifact.create(
        kind="selective_policy", fit_artifact_id=artifact.artifact_id, contract=artifact.contract, scope=artifact.scope,
        fit_source=artifact.source, validation_source=provenance, threshold=selected,
        risk_target=settings.risk_target, min_accepted=settings.min_accepted, criterion=settings.criterion,
        candidates=candidates, candidate_count=len(candidates), alpha=settings.alpha,
        assume_iid_observations=settings.assume_iid_observations,
        selection=SelectionEvidence(
            accepted_count=operating.accepted_count, error_count=operating.error_count,
            risk=operating.risk, coverage=operating.coverage,
        ), selection_upper_risk=upper_risk, reason="feasible_threshold" if selected is not None else "no_feasible_threshold",
        guarantee=guarantee, overrides=calibrated.overrides,
    )
