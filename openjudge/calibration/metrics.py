# -*- coding: utf-8 -*-
"""Reliability metrics and descriptive group-bootstrap uncertainty."""

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import Field, field_validator
from sklearn.metrics import roc_auc_score

from openjudge.calibration.contracts import FrozenModel, Observation, ObservationBatch

DEFAULT_BIN_EDGES = tuple(index / 10 for index in range(11))


class ReliabilityBin(FrozenModel):
    """One left-closed bin; the final bin also includes probability one."""

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    absolute_gap: float | None


class ReliabilitySummary(FrozenModel):
    """Calibration, decision accuracy and discrimination are distinct quantities."""

    sample_count: int
    accuracy: float | None
    ece: float | None
    mce: float | None
    brier: float | None
    log_loss: float | None
    roc_auc: float | None


class ReliabilityMetrics(ReliabilitySummary):
    """Summary plus inspectable bin support and complete input denominators."""

    bins: tuple[ReliabilityBin, ...]
    bin_edges: tuple[float, ...]
    log_loss_clip: float
    total_count: int
    status_counts: dict[str, int]
    domains: dict[str, ReliabilitySummary]


class BootstrapSettings(FrozenModel):
    """Local seeded percentile bootstrap; never a selection risk guarantee."""

    resamples: int = Field(default=200, ge=0, le=100000, strict=True)
    seed: int = Field(default=0, ge=0, strict=True)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @field_validator("confidence_level", mode="before")
    @classmethod
    def finite_level(cls, value: float) -> float:
        """Reject boolean probabilities in uncertainty configuration."""
        if isinstance(value, bool):
            raise ValueError("confidence level must be a probability")
        return value


class PercentileInterval(FrozenModel):
    """Undefined replicates are omitted and their effective count disclosed."""

    lower: float | None
    upper: float | None
    effective_resamples: int


class BootstrapResult(FrozenModel):
    """Descriptive sampling variability under whole-group resampling."""

    group_count: int
    resamples: int
    seed: int
    confidence_level: float
    intervals: dict[str, PercentileInterval]
    interpretation: str = (
        "descriptive percentile intervals from whole-group resampling; groups must represent independent "
        "sampling units for a sampling interpretation; no population-risk or distribution-shift guarantee"
    )


def validate_metric_settings(bin_edges: Sequence[float], log_loss_clip: float) -> tuple[float, ...]:
    """Validate endpoints and clipping without repairing invalid configurations."""
    if isinstance(log_loss_clip, bool) or not math.isfinite(log_loss_clip) or not 0 < log_loss_clip < 0.5:
        raise ValueError("log_loss_clip must lie strictly between 0 and 0.5")
    edges = tuple(float(edge) for edge in bin_edges)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1:
        raise ValueError("bin edges must start at 0 and end at 1")
    if any(not math.isfinite(edge) for edge in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("bin edges must be finite and strictly increasing")
    return edges


def _summarize(
    rows: Sequence[Observation],
    edges: tuple[float, ...],
    clip: float,
    *,
    discrimination: bool = True,
) -> tuple[ReliabilitySummary, tuple[ReliabilityBin, ...]]:
    """Compute pure numerical summaries; never inspect labels or native results."""
    valid = [row for row in rows if row.status == "valid"]
    probabilities = np.asarray([row.probability for row in valid], dtype=float)
    outcomes = np.asarray([row.correctness for row in valid], dtype=float)
    assignments = np.searchsorted(edges, probabilities, side="right") - 1
    assignments = np.minimum(assignments, len(edges) - 2)
    bins = []
    weighted_gaps = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        mask = assignments == index
        count = int(np.count_nonzero(mask))
        confidence = float(np.mean(probabilities[mask])) if count else None
        accuracy = float(np.mean(outcomes[mask])) if count else None
        gap = abs(confidence - accuracy) if count else None
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )
        if count:
            weighted_gaps.append((count, gap))
    count = len(valid)
    clipped = np.clip(probabilities, clip, 1 - clip)
    auc = None
    if discrimination and count and len(set(outcomes)) == 2:
        auc = float(roc_auc_score(outcomes, probabilities))
    summary = ReliabilitySummary(
        sample_count=count,
        accuracy=float(np.mean(outcomes)) if count else None,
        ece=sum(n * gap for n, gap in weighted_gaps) / count if count else None,
        mce=max((gap for _, gap in weighted_gaps), default=None),
        brier=float(np.mean((probabilities - outcomes) ** 2)) if count else None,
        log_loss=float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log1p(-clipped))) if count else None,
        roc_auc=auc,
    )
    return summary, tuple(bins)


def reliability_metrics(
    batch: ObservationBatch,
    *,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    log_loss_clip: float = 1e-15,
) -> ReliabilityMetrics:
    """Evaluate confidence calibration without fitting or changing observations.

    Args:
        batch: Ordered correctness-confidence observations with omission statuses.
        bin_edges: Fixed-width default or explicit increasing edges from 0 to 1.
        log_loss_clip: Endpoint protection for loss only, disclosed in the result.

    Returns:
        Bin support, ECE/MCE, Brier, clipped log loss, accuracy, ROC AUC and domains.
        All undefined metrics are None, including empty-data accuracy.
    """
    edges = validate_metric_settings(bin_edges, log_loss_clip)
    summary, bins = _summarize(batch.observations, edges, log_loss_clip)
    domains = defaultdict(list)
    for row in batch.observations:
        if row.domain is not None:
            domains[row.domain].append(row)
    return ReliabilityMetrics(
        **summary.model_dump(),
        bins=bins,
        bin_edges=edges,
        log_loss_clip=log_loss_clip,
        total_count=len(batch.observations),
        status_counts=batch.counts,
        domains={name: _summarize(rows, edges, log_loss_clip)[0] for name, rows in sorted(domains.items())},
    )


def group_bootstrap(
    batch: ObservationBatch,
    settings: BootstrapSettings = BootstrapSettings(),
    *,
    threshold: float | None = None,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    log_loss_clip: float = 1e-15,
) -> BootstrapResult:
    """Resample complete groups using an isolated RNG and fixed fitted decisions.

    Args:
        batch: Includes invalid/error rows so coverage denominators are resampled.
        settings: Number of replicates, deterministic seed and percentile level.
        threshold: Fixed acceptance threshold; None represents abstain-all.
        bin_edges: Same fixed bins as the point estimate.
        log_loss_clip: Same clipping as the point estimate.

    Returns:
        Descriptive intervals with effective counts for each defined statistic.
        Neither calibrator nor threshold is refitted in any bootstrap replicate.
    """
    edges = validate_metric_settings(bin_edges, log_loss_clip)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must lie in [0, 1], or None for abstain-all")
    groups = defaultdict(list)
    for row in batch.observations:
        groups[row.group_id].append(row)
    ordered_groups = [groups[key] for key in sorted(groups)]
    samples: dict[str, list[float]] = {
        key: [] for key in ("accuracy", "ece", "brier", "log_loss", "accepted_risk", "coverage")
    }
    rng = np.random.default_rng(settings.seed)
    if ordered_groups:
        for _ in range(settings.resamples):
            selected = rng.integers(0, len(ordered_groups), size=len(ordered_groups))
            rows = [row for index in selected for row in ordered_groups[index]]
            summary, _ = _summarize(rows, edges, log_loss_clip, discrimination=False)
            for key in ("accuracy", "ece", "brier", "log_loss"):
                value = getattr(summary, key)
                if value is not None:
                    samples[key].append(value)
            accepted = [r for r in rows if r.status == "valid" and threshold is not None and r.probability >= threshold]
            samples["coverage"].append(len(accepted) / len(rows))
            if accepted:
                samples["accepted_risk"].append(sum(1 - r.correctness for r in accepted) / len(accepted))
    tail = (1 - settings.confidence_level) / 2
    intervals = {}
    for key, values in samples.items():
        endpoints = np.quantile(values, [tail, 1 - tail]) if values else (None, None)
        intervals[key] = PercentileInterval(
            lower=float(endpoints[0]) if values else None,
            upper=float(endpoints[1]) if values else None,
            effective_resamples=len(values),
        )
    return BootstrapResult(
        group_count=len(groups),
        resamples=settings.resamples,
        seed=settings.seed,
        confidence_level=settings.confidence_level,
        intervals=intervals,
    )
