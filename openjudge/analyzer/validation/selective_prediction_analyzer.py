# -*- coding: utf-8 -*-
"""Native fixed-policy selective evaluation; never optimize on analyzed labels."""

from typing import Any, List

from openjudge.analyzer.base_analyzer import AnalysisResult
from openjudge.analyzer.validation.base_validation_analyzer import (
    BaseValidationAnalyzer,
)
from openjudge.calibration.adapter import DecisionAdapter
from openjudge.calibration.artifacts import FitArtifact, PolicyArtifact
from openjudge.calibration.contracts import (
    CalibrationContract,
    CalibrationScope,
    probability,
)
from openjudge.calibration.extraction import ConfidenceExtractor, extract_observations
from openjudge.calibration.fitting import apply_calibrator
from openjudge.calibration.metrics import (
    BootstrapResult,
    BootstrapSettings,
    group_bootstrap,
)
from openjudge.calibration.selective import (
    RiskCoverageCurve,
    SelectiveMetrics,
    risk_coverage_curve,
    selective_metrics,
)
from openjudge.graders.schema import GraderResult


class SelectivePredictionAnalysisResult(AnalysisResult):
    """Accepted risk, all-input accounting and complete-tie risk/coverage curve."""

    operating: SelectiveMetrics
    curve: RiskCoverageCurve
    bootstrap: BootstrapResult | None = None


class SelectivePredictionAnalyzer(BaseValidationAnalyzer):
    """Evaluate a fixed threshold or saved policy on native grader results.

    Args:
        contract: Explicit event and confidence contract.
        scope: Actual judge/model/application identity.
        threshold: Fixed acceptance threshold; None means abstain-all.
        artifact: Optional independently fitted calibrator.
        policy: Optional saved threshold policy; requires its bound artifact.
        invalid_policy: Strict validation or visible exclusions.
        confidence_extractor: Explicit named-contract callback.
        bootstrap: Optional descriptive whole-group intervals for fixed decisions.
    """

    name = "Selective Prediction Analysis"

    def __init__(
        self,
        contract: CalibrationContract,
        scope: CalibrationScope,
        *,
        threshold: float | None = 0.0,
        artifact: FitArtifact | None = None,
        policy: PolicyArtifact | None = None,
        invalid_policy: str = "raise",
        confidence_extractor: ConfidenceExtractor | None = None,
        bootstrap: BootstrapSettings | None = None,
    ) -> None:
        if policy is not None:
            if artifact is None:
                raise ValueError("saved policy analysis requires its fitted artifact")
            DecisionAdapter(artifact, policy, contract, scope, confidence_extractor=confidence_extractor)
            threshold = policy.threshold
        self.threshold = probability(threshold) if threshold is not None else None
        self.contract = contract
        self.scope = scope
        self.artifact = artifact
        self.policy = policy
        self.invalid_policy = invalid_policy
        self.confidence_extractor = confidence_extractor
        self.bootstrap_settings = bootstrap

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        label_path: str | None = None,
        **kwargs: Any,
    ) -> SelectivePredictionAnalysisResult:
        """Evaluate fixed decisions, retaining native usable-result coverage separately."""
        self._validate_alignment(dataset, grader_results)
        if kwargs:
            raise ValueError("configure selective settings in the analyzer constructor")
        if label_path is not None and label_path != self.contract.label_path:
            raise ValueError("label_path must match the explicit calibration contract")
        batch = extract_observations(
            dataset,
            grader_results,
            self.contract,
            self.scope,
            invalid_policy=self.invalid_policy,
            confidence_extractor=self.confidence_extractor,
        )
        if self.artifact is not None:
            batch = apply_calibrator(self.artifact, batch)
        uncertainty = (
            group_bootstrap(batch, self.bootstrap_settings, threshold=self.threshold)
            if self.bootstrap_settings
            else None
        )
        return SelectivePredictionAnalysisResult(
            name=self.name,
            operating=selective_metrics(batch, self.threshold),
            curve=risk_coverage_curve(batch),
            bootstrap=uncertainty,
            metadata={
                **self._coverage_metadata(grader_results, len(batch.valid), "Risk of fixed selective decisions"),
                "target_id": self.contract.target_id,
                "contract_fingerprint": self.contract.fingerprint,
                "scope": self.scope.model_dump(),
                "fit_artifact_id": batch.fit_artifact_id,
                "policy_id": self.policy.artifact_id if self.policy is not None else None,
                "status_counts": batch.counts,
            },
        )
