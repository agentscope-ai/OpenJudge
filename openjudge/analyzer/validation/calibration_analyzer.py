# -*- coding: utf-8 -*-
"""Native validation analyzer for explicit correctness-confidence reliability."""

from typing import Any, List, Sequence

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import BaseValidationAnalyzer
from openjudge.calibration.artifacts import FitArtifact
from openjudge.calibration.contracts import CalibrationContract, CalibrationScope
from openjudge.calibration.extraction import ConfidenceExtractor, extract_observations
from openjudge.calibration.fitting import apply_calibrator
from openjudge.calibration.metrics import (
    DEFAULT_BIN_EDGES, BootstrapResult, BootstrapSettings, ReliabilityMetrics, group_bootstrap, reliability_metrics,
)
from openjudge.graders.schema import GraderResult


class CalibrationAnalysisResult(AnalysisResult):
    """Reliability and optional descriptive uncertainty with native coverage metadata."""

    reliability: ReliabilityMetrics
    bootstrap: BootstrapResult | None = None


class CalibrationAnalyzer(BaseValidationAnalyzer):
    """Analyze declared confidence; an optional artifact is applied, never fitted.

    Args:
        contract: Explicit correctness event, decisions and confidence extraction.
        scope: Actual judge/model/application identity.
        artifact: Optional independently fitted postprocessor.
        invalid_policy: Strict by default; exclude records each malformed row.
        confidence_extractor: Callback matching a declared stable extractor ID.
        bin_edges: Reliability bins fixed before analysis.
        log_loss_clip: Disclosed clipping applied only inside binary log loss.
        bootstrap: Optional descriptive group-resampling settings.
    """

    name = "Calibration Analysis"

    def __init__(
        self, contract: CalibrationContract, scope: CalibrationScope, *, artifact: FitArtifact | None = None,
        invalid_policy: str = "raise", confidence_extractor: ConfidenceExtractor | None = None,
        bin_edges: Sequence[float] = DEFAULT_BIN_EDGES, log_loss_clip: float = 1e-15,
        bootstrap: BootstrapSettings | None = None,
    ) -> None:
        self.contract = contract
        self.scope = scope
        self.artifact = artifact
        self.invalid_policy = invalid_policy
        self.confidence_extractor = confidence_extractor
        self.bin_edges = tuple(bin_edges)
        self.log_loss_clip = log_loss_clip
        self.bootstrap_settings = bootstrap

    def analyze(
        self, dataset: List[dict], grader_results: List[GraderResult], label_path: str | None = None, **kwargs: Any,
    ) -> CalibrationAnalysisResult:
        """Compute native aligned reliability; labels must match the declared contract."""
        self._validate_alignment(dataset, grader_results)
        if kwargs:
            raise ValueError("configure calibration settings in the analyzer constructor")
        if label_path is not None and label_path != self.contract.label_path:
            raise ValueError("label_path must match the explicit calibration contract")
        batch = extract_observations(
            dataset, grader_results, self.contract, self.scope, invalid_policy=self.invalid_policy,
            confidence_extractor=self.confidence_extractor,
        )
        if self.artifact is not None:
            batch = apply_calibrator(self.artifact, batch)
        result = reliability_metrics(batch, bin_edges=self.bin_edges, log_loss_clip=self.log_loss_clip)
        uncertainty = None
        if self.bootstrap_settings is not None:
            uncertainty = group_bootstrap(
                batch, self.bootstrap_settings, bin_edges=self.bin_edges, log_loss_clip=self.log_loss_clip,
            )
        return CalibrationAnalysisResult(
            name=self.name, reliability=result, bootstrap=uncertainty, metadata={
                **self._coverage_metadata(grader_results, len(batch.valid), "Reliability of declared correctness confidence"),
                "target_id": self.contract.target_id, "contract_fingerprint": self.contract.fingerprint,
                "scope": self.scope.model_dump(), "fit_artifact_id": batch.fit_artifact_id, "status_counts": batch.counts,
            },
        )
