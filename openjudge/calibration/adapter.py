# -*- coding: utf-8 -*-
"""Opt-in native decision envelopes; never replace scores or request more calls."""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import SerializeAsAny

from openjudge.calibration.artifacts import (
    FitArtifact,
    PolicyArtifact,
    check_compatibility,
)
from openjudge.calibration.contracts import (
    CalibrationContract,
    CalibrationScope,
    CompatibilityOverride,
    FrozenModel,
    is_decision_label,
)
from openjudge.calibration.extraction import (
    ConfidenceExtractor,
    extract_confidence,
    read_path,
)
from openjudge.calibration.fitting import predict_confidence
from openjudge.graders.schema import GraderError, GraderResult, GraderScore


class DecisionEnvelope(FrozenModel):
    """Explicit availability/acceptance alongside the unmodified native result."""

    status: Literal["accepted", "abstained", "unavailable"]
    original_result: SerializeAsAny[GraderResult]
    raw_confidence: float | None
    calibrated_confidence: float | None
    fit_artifact_id: str
    policy_id: str
    threshold: float | None
    reason: str
    overrides: tuple[CompatibilityOverride, ...] = ()


class DecisionAdapter:
    """Apply a fixed policy without labels, grading, routing or score mutation.

    Args:
        artifact: Independently fitted correctness-confidence postprocessor.
        policy: Validation-selected policy bound to that exact fitted artifact.
        contract: Actual inference event/extractor configuration.
        scope: Actual inference judge/model/application identity.
        confidence_extractor: Callback for the contract's explicit extractor ID.
        override_reason: Recorded intentional transfer across compatibility boundaries.
    """

    def __init__(
        self,
        artifact: FitArtifact,
        policy: PolicyArtifact,
        contract: CalibrationContract,
        scope: CalibrationScope,
        *,
        confidence_extractor: ConfidenceExtractor | None = None,
        override_reason: str | None = None,
    ) -> None:
        if policy.fit_artifact_id != artifact.artifact_id or policy.fit_source != artifact.source:
            raise ValueError("policy is bound to a different fitted calibrator")
        if policy.contract != artifact.contract or policy.scope != artifact.scope:
            raise ValueError("policy and fitted artifact identities disagree")
        if (contract.extractor_id is not None) != (confidence_extractor is not None):
            raise ValueError("declared confidence extractor and supplied callback must match")
        override = check_compatibility(contract, scope, artifact.contract, artifact.scope, override_reason)
        self.artifact = artifact
        self.policy = policy
        self.contract = contract
        self.scope = scope
        self.confidence_extractor = confidence_extractor
        self.overrides = policy.overrides + ((override,) if override is not None else ())

    def decide(self, result: GraderResult, sample: dict[str, Any] | None = None) -> DecisionEnvelope:
        """Annotate one native result; no independent target label is read at inference.

        Args:
            result: Original native result, retained by reference.
            sample: Needed only for a declared dataset confidence path or callback.

        Returns:
            Accepted/abstained on valid confidence, unavailable on provider or
            input failure. Missing confidence remains visible even for abstain-all.
        """
        sample = {} if sample is None else sample
        raw = None
        calibrated = None
        reason = None
        if isinstance(result, GraderError):
            reason = "provider_error"
        elif not isinstance(result, GraderScore):
            reason = "unsupported_result"
        elif self.contract.exclude_path is not None and read_path(sample, self.contract.exclude_path) is True:
            reason = "user_excluded"
        else:
            prediction = read_path(result, self.contract.prediction_path)
            if prediction is None:
                reason = "missing_prediction"
            elif not is_decision_label(prediction) or prediction not in self.contract.decision_labels:
                reason = "invalid_prediction"
            else:
                raw, reason = extract_confidence(sample, result, self.contract, self.confidence_extractor)
        if reason is not None:
            status = "unavailable"
        else:
            calibrated = predict_confidence(self.artifact, [raw])[0]
            if self.policy.threshold is not None and calibrated >= self.policy.threshold:
                status = "accepted"
                reason = "confidence_at_or_above_threshold"
            else:
                status = "abstained"
                reason = "abstain_all_policy" if self.policy.threshold is None else "confidence_below_threshold"
        return DecisionEnvelope(
            status=status,
            original_result=result,
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            fit_artifact_id=self.artifact.artifact_id,
            policy_id=self.policy.artifact_id,
            threshold=self.policy.threshold,
            reason=reason,
            overrides=self.overrides,
        )

    def decide_many(
        self,
        grader_results: Sequence[GraderResult],
        dataset: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[DecisionEnvelope, ...]:
        """Preserve runner order; mismatched sample/result lists fail before inference."""
        if dataset is None:
            dataset = [{} for _ in grader_results]
        if len(dataset) != len(grader_results):
            raise ValueError("dataset and grader_results must have the same length")
        return tuple(self.decide(result, sample) for sample, result in zip(dataset, grader_results))
