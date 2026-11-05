# -*- coding: utf-8 -*-
"""Separated fit/select/test orchestration and summary-only reproducible reports."""

from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

from openjudge.calibration.artifacts import (
    FitArtifact,
    FitSettings,
    JSONArtifact,
    PolicyArtifact,
    check_compatibility,
)
from openjudge.calibration.cache import CachedPartition
from openjudge.calibration.contracts import (
    CalibrationContract,
    CalibrationScope,
    FrozenModel,
    ObservationBatch,
)
from openjudge.calibration.extraction import ConfidenceExtractor, extract_observations
from openjudge.calibration.fitting import apply_calibrator, fit_calibrator
from openjudge.calibration.metrics import (
    DEFAULT_BIN_EDGES,
    BootstrapResult,
    BootstrapSettings,
    ReliabilityMetrics,
    group_bootstrap,
    reliability_metrics,
    validate_metric_settings,
)
from openjudge.calibration.selective import (
    RiskCoverageCurve,
    SelectionSettings,
    SelectiveMetrics,
    risk_coverage_curve,
    select_policy,
    selective_metrics,
)
from openjudge.calibration.splits import (
    SplitManifest,
    assert_disjoint,
    audit_partitions,
    partition_provenance,
)


class WorkflowSettings(FrozenModel):
    """One explicit, serializable offline experiment configuration."""

    fit: FitSettings = Field(default_factory=FitSettings)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES
    log_loss_clip: float = 1e-15
    invalid_policy: Literal["raise", "exclude"] = "raise"

    @model_validator(mode="after")
    def metric_configuration(self) -> "WorkflowSettings":
        """Validate shared metric choices before processing any partition."""
        validate_metric_settings(self.bin_edges, self.log_loss_clip)
        return self


class PartitionSummary(FrozenModel):
    """Support and input statuses only; no per-row labels or prompts."""

    role: str
    total_count: int
    valid_count: int
    group_count: int
    status_counts: dict[str, int]
    domain_counts: dict[str, int]


class ConfidenceEvaluation(FrozenModel):
    """Fixed-confidence reliability, complete-tie curve and descriptive intervals."""

    reliability: ReliabilityMetrics
    curve: RiskCoverageCurve
    bootstrap: BootstrapResult


class TestEvaluation(FrozenModel):
    """Held-out results with separate raw/calibrated and acceptance accounting."""

    raw: ConfidenceEvaluation
    calibrated: ConfidenceEvaluation
    selected: SelectiveMetrics
    always_accept: SelectiveMetrics
    selected_bootstrap: BootstrapResult


class DomainDiagnostics(FrozenModel):
    """Observed support differences and extrapolation; no claim of solved shift."""

    calibration_domains: dict[str, int]
    validation_domains: dict[str, int]
    test_domains: dict[str, int]
    unseen_test_domains: tuple[str, ...]
    test_confidences_outside_fit_range: int
    interpretation: str = (
        "domain labels are caller-supplied; missing labels hide possible shift; support/extrapolation "
        "diagnostics do not establish exchangeability or guarantee risk under distribution shift"
    )


class WorkflowReport(JSONArtifact):
    """Content-addressed report and safe artifacts from a single separated experiment."""

    kind: Literal["correctness_calibration_report"] = "correctness_calibration_report"
    settings: WorkflowSettings
    manifest: SplitManifest
    partitions: tuple[PartitionSummary, ...]
    fit_artifact: FitArtifact
    policy: PolicyArtifact
    test: TestEvaluation
    domains: DomainDiagnostics
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def coherent_experiment(self) -> "WorkflowReport":
        """Recheck artifact/source binding when reloading a cached report."""
        roles = tuple(part.role for part in self.manifest.partitions)
        if roles != ("calibration", "validation", "test"):
            raise ValueError("report manifest must separate calibration, validation and test")
        if self.fit_artifact.source != self.manifest.partitions[0]:
            raise ValueError("report fit source does not match its manifest")
        if self.policy.fit_artifact_id != self.fit_artifact.artifact_id:
            raise ValueError("report policy does not reference its fitted artifact")
        if self.policy.validation_source != self.manifest.partitions[1]:
            raise ValueError("report policy source does not match validation manifest")
        return self


def _partition_summary(batch: ObservationBatch) -> PartitionSummary:
    """Keep statuses for every row and per-domain valid support."""
    return PartitionSummary(
        role=batch.role,
        total_count=len(batch.observations),
        valid_count=len(batch.valid),
        group_count=len({row.group_id for row in batch.observations}),
        status_counts=batch.counts,
        domain_counts=dict(Counter(row.domain for row in batch.valid if row.domain is not None)),
    )


def _confidence_evaluation(batch: ObservationBatch, settings: WorkflowSettings) -> ConfidenceEvaluation:
    """Compute summaries against fixed inputs without fitting or selecting anything."""
    return ConfidenceEvaluation(
        reliability=reliability_metrics(batch, bin_edges=settings.bin_edges, log_loss_clip=settings.log_loss_clip),
        curve=risk_coverage_curve(batch),
        bootstrap=group_bootstrap(
            batch,
            settings.bootstrap,
            bin_edges=settings.bin_edges,
            log_loss_clip=settings.log_loss_clip,
        ),
    )


def evaluate_policy(
    test: ObservationBatch,
    artifact: FitArtifact,
    policy: PolicyArtifact,
    settings: WorkflowSettings = WorkflowSettings(),
) -> TestEvaluation:
    """Evaluate one fixed calibrator/policy against a disjoint test partition.

    Args:
        test: Raw held-out observations; its labels never reach fitting or selection.
        artifact: Independently fitted postprocessor.
        policy: Separately validation-selected threshold.
        settings: Fixed metric bins, clipping and descriptive uncertainty settings.

    Returns:
        Raw and calibrated held-out reliability, risk/coverage and operating points.
        No metric in this function changes the fixed fit or threshold artifacts.
    """
    if test.role != "test":
        raise ValueError("evaluate_policy requires a test partition")
    if policy.fit_artifact_id != artifact.artifact_id or policy.fit_source != artifact.source:
        raise ValueError("policy is bound to a different fitted artifact")
    check_compatibility(policy.contract, policy.scope, artifact.contract, artifact.scope)
    assert_disjoint(artifact.source, policy.validation_source, partition_provenance(test))
    calibrated = apply_calibrator(artifact, test)
    return TestEvaluation(
        raw=_confidence_evaluation(test, settings),
        calibrated=_confidence_evaluation(calibrated, settings),
        selected=selective_metrics(calibrated, policy.threshold),
        always_accept=selective_metrics(calibrated, 0),
        selected_bootstrap=group_bootstrap(
            calibrated,
            settings.bootstrap,
            threshold=policy.threshold,
            bin_edges=settings.bin_edges,
            log_loss_clip=settings.log_loss_clip,
        ),
    )


def run_calibration_workflow(
    calibration: CachedPartition,
    validation: CachedPartition,
    test: CachedPartition,
    contract: CalibrationContract,
    scope: CalibrationScope,
    settings: WorkflowSettings = WorkflowSettings(),
    *,
    confidence_extractor: ConfidenceExtractor | None = None,
) -> WorkflowReport:
    """Audit all partitions, fit calibration only, select validation only, then test.

    Args:
        calibration: Native cached results with the calibration role.
        validation: Native cached results with the threshold-validation role.
        test: Native cached results with the held-out final-test role.
        contract: Binary correctness confidence event and extraction fields.
        scope: Versioned model/judge/application identity shared by this experiment.
        settings: Explicit postprocessor, selection and evaluation settings.
        confidence_extractor: Optional callback named by the confidence contract.

    Returns:
        A reproducible summary report with fitted/selected primitive JSON artifacts.
        This orchestration cannot prevent a caller repeatedly inspecting test data;
        the caller must preserve the predeclared held-out experimental protocol.
    """
    inputs = (calibration, validation, test)
    if tuple(part.role for part in inputs) != ("calibration", "validation", "test"):
        raise ValueError("workflow requires separate calibration, validation and test roles in that order")
    batches = tuple(
        extract_observations(
            part.dataset,
            part.grader_results,
            contract,
            scope,
            role=part.role,
            invalid_policy=settings.invalid_policy,
            confidence_extractor=confidence_extractor,
        )
        for part in inputs
    )
    manifest = audit_partitions(*batches)
    # Fit and select receive only their own batches, never the held-out labels.
    artifact = fit_calibrator(batches[0], settings.fit)
    policy = select_policy(batches[1], artifact, settings.selection)
    evaluation = evaluate_policy(batches[2], artifact, policy, settings)
    summaries = tuple(_partition_summary(batch) for batch in batches)
    fit_domains, val_domains, test_domains = (summary.domain_counts for summary in summaries)
    low, high = artifact.confidence_range
    diagnostics = DomainDiagnostics(
        calibration_domains=fit_domains,
        validation_domains=val_domains,
        test_domains=test_domains,
        unseen_test_domains=tuple(sorted(set(test_domains) - set(fit_domains))),
        test_confidences_outside_fit_range=sum(not low <= row.probability <= high for row in batches[2].valid),
    )
    return WorkflowReport.create(
        kind="correctness_calibration_report",
        settings=settings,
        manifest=manifest,
        partitions=summaries,
        fit_artifact=artifact,
        policy=policy,
        test=evaluation,
        domains=diagnostics,
        limitations=(
            "Correctness labels must be independently supplied; this workflow does not establish their quality.",
            "Strict group/sample disjointness cannot discover undeclared duplicate tasks or prompt-selection leakage.",
            "Empirical threshold selection and group-bootstrap intervals carry no population-risk guarantee.",
            "Hoeffding interpretation requires all stated fixed-grid and i.i.d. assumptions, and does not cover shift.",
            "Synthetic/offline engineering validation is not evidence of improved real-agent performance.",
            "The caller must keep final-test data held out and disclose repeated inspection or experiment selection.",
        ),
    )
