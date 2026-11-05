# -*- coding: utf-8 -*-
"""Offline calibration of explicitly declared binary judge-correctness confidence."""

from .adapter import DecisionAdapter, DecisionEnvelope
from .artifacts import FitArtifact, PolicyArtifact
from .cache import CachedPartition
from .contracts import (
    CalibrationContract,
    CalibrationScope,
    Observation,
    ObservationBatch,
)
from .extraction import extract_observations, vote_fraction
from .fitting import FitSettings, apply_calibrator, fit_calibrator, predict_confidence
from .metrics import BootstrapSettings, group_bootstrap, reliability_metrics
from .selective import (
    SelectionSettings,
    risk_coverage_curve,
    select_policy,
    selective_metrics,
)
from .splits import SplitManifest, audit_partitions
from .workflow import (
    WorkflowReport,
    WorkflowSettings,
    evaluate_policy,
    run_calibration_workflow,
)

__all__ = [
    "BootstrapSettings",
    "CachedPartition",
    "CalibrationContract",
    "CalibrationScope",
    "DecisionAdapter",
    "DecisionEnvelope",
    "FitArtifact",
    "FitSettings",
    "Observation",
    "ObservationBatch",
    "PolicyArtifact",
    "SelectionSettings",
    "SplitManifest",
    "WorkflowReport",
    "WorkflowSettings",
    "apply_calibrator",
    "audit_partitions",
    "evaluate_policy",
    "extract_observations",
    "fit_calibrator",
    "group_bootstrap",
    "predict_confidence",
    "reliability_metrics",
    "risk_coverage_curve",
    "run_calibration_workflow",
    "select_policy",
    "selective_metrics",
    "vote_fraction",
]
