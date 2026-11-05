# -*- coding: utf-8 -*-
"""Fit monotone confidence postprocessors on explicit calibration data only."""

import warnings
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from openjudge.calibration.artifacts import FitArtifact, FitSettings, FitState, check_compatibility
from openjudge.calibration.contracts import ObservationBatch, probability
from openjudge.calibration.splits import partition_provenance


def fit_calibrator(batch: ObservationBatch, settings: FitSettings = FitSettings()) -> FitArtifact:
    """Fit identity, a nonnegative sigmoid on raw probabilities, or isotonic regression.

    Args:
        batch: Calibration-role observations with independent correctness labels.
        settings: Existing sklearn-compatible estimation settings.

    Returns:
        A content-addressed JSON artifact, containing no pickled estimator.
        Constant inputs/outcomes and negative unconstrained sigmoid slopes have
        explicit constant states. No test or validation partition is accepted.
    """
    if batch.role != "calibration":
        raise ValueError("fit_calibrator requires a calibration partition")
    if batch.fit_artifact_id is not None:
        raise ValueError("fit_calibrator requires raw, previously uncalibrated confidence")
    rows = batch.valid
    if not rows:
        raise ValueError("fitting requires valid calibration observations")
    probabilities = np.asarray([row.probability for row in rows], dtype=float)
    outcomes = np.asarray([row.correctness for row in rows], dtype=int)
    degeneracy = None
    if settings.method == "identity":
        state = FitState(kind="identity")
    elif len(set(probabilities)) == 1 or len(set(outcomes)) == 1:
        state = FitState(kind="constant", constant=float(np.mean(outcomes)))
        degeneracy = "constant_confidence" if len(set(probabilities)) == 1 else "constant_outcome"
    elif settings.method == "isotonic":
        estimator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        estimator.fit(probabilities, outcomes)
        state = FitState(
            kind="isotonic", x_thresholds=tuple(estimator.X_thresholds_), y_thresholds=tuple(estimator.y_thresholds_),
        )
    else:
        estimator = LogisticRegression(
            C=settings.regularization, solver="lbfgs", max_iter=settings.max_iterations, tol=settings.tolerance,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                estimator.fit(probabilities.reshape(-1, 1), outcomes)
            except ConvergenceWarning as exc:
                raise ValueError("sigmoid fit did not converge; increase max_iterations or inspect data") from exc
        slope = float(estimator.coef_[0, 0])
        if slope < 0:
            # Convex logistic loss with nonnegative slope: boundary optimum has
            # slope zero and unpenalized intercept equal to logit(mean outcome).
            state = FitState(kind="constant", constant=float(np.mean(outcomes)))
            degeneracy = "nonpositive_monotone_relationship"
        else:
            state = FitState(kind="sigmoid", slope=slope, intercept=float(estimator.intercept_[0]))
    return FitArtifact.create(
        kind="calibrator", contract=batch.contract, scope=batch.scope, settings=settings, state=state,
        source=partition_provenance(batch), correct_count=int(np.sum(outcomes)),
        confidence_range=(float(np.min(probabilities)), float(np.max(probabilities))),
        domain_counts=dict(Counter(row.domain for row in rows if row.domain is not None)), degeneracy=degeneracy,
        implementation="openjudge.correctness-calibration.v1",
    )


def predict_confidence(artifact: FitArtifact, probabilities: Sequence[float]) -> tuple[float, ...]:
    """Apply primitive fitted state; invalid raw inputs never become clipped probabilities."""
    values = np.asarray([probability(value) for value in probabilities], dtype=float)
    state = artifact.state
    if state.kind == "identity":
        result = values
    elif state.kind == "constant":
        result = np.full(values.shape, state.constant, dtype=float)
    elif state.kind == "isotonic":
        result = np.interp(values, state.x_thresholds, state.y_thresholds)
    else:
        logits = state.slope * values + state.intercept
        result = np.empty_like(logits)
        positive = logits >= 0
        result[positive] = 1 / (1 + np.exp(-logits[positive]))
        exponentials = np.exp(logits[~positive])
        result[~positive] = exponentials / (1 + exponentials)
    return tuple(float(value) for value in result)


def apply_calibrator(
    artifact: FitArtifact, batch: ObservationBatch, *, override_reason: str | None = None,
) -> ObservationBatch:
    """Return calibrated copies while retaining all rows, identities and omissions.

    Args:
        artifact: Already fitted confidence postprocessor.
        batch: Any explicit partition with compatible target/extraction/scope.
        override_reason: Optional recorded explanation for intentional transfer.

    Returns:
        A new immutable batch marked with the fitted artifact ID. Double
        calibration fails rather than silently applying a nonlinear map twice.
    """
    if batch.fit_artifact_id is not None:
        raise ValueError("confidence batch has already been calibrated")
    override = check_compatibility(batch.contract, batch.scope, artifact.contract, artifact.scope, override_reason)
    predictions = iter(predict_confidence(artifact, [row.probability for row in batch.valid]))
    observations = tuple(
        row.model_copy(update={"probability": next(predictions)}) if row.status == "valid" else row
        for row in batch.observations
    )
    return batch.model_copy(update={
        "observations": observations, "fit_artifact_id": artifact.artifact_id,
        "overrides": batch.overrides + ((override,) if override is not None else ()),
    })
