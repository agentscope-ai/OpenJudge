"""Fit only calibration data and persist validated numerical state."""

import json

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from openjudge.calibration.artifacts import FitArtifact
from openjudge.calibration.contracts import CalibrationScope
from openjudge.calibration.fitting import (
    FitSettings,
    apply_calibrator,
    fit_calibrator,
    predict_confidence,
)


def test_identity_preserves_endpoints_and_artifact_round_trip(make_batch):
    batch = make_batch([0, 0.2, 1], [0, 1, 1])
    fit = fit_calibrator(batch, FitSettings(method="identity"))
    loaded = FitArtifact.model_validate_json(fit.model_dump_json())
    assert predict_confidence(loaded, [0, 0.2, 1]) == (0.0, 0.2, 1.0)
    assert fit.artifact_id == loaded.artifact_id
    assert fit.source.role == "calibration"
    assert fit.source.sample_count == 3
    assert apply_calibrator(fit, batch).observations == batch.observations


def test_isotonic_matches_sklearn_interpolation(make_batch):
    probabilities = [0.1, 0.1, 0.2, 0.4, 0.8, 0.9]
    outcomes = [0, 1, 0, 1, 0, 1]
    fit = fit_calibrator(make_batch(probabilities, outcomes), FitSettings(method="isotonic"))
    reference = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(probabilities, outcomes)
    grid = np.linspace(0, 1, 31)
    assert predict_confidence(fit, grid) == pytest.approx(reference.predict(grid))
    assert all(a <= b for a, b in zip(predict_confidence(fit, grid), predict_confidence(fit, grid)[1:]))


def test_sigmoid_matches_reference_on_raw_confidence(make_batch):
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    outcomes = [0, 0, 1, 0, 1, 0, 1, 1]
    settings = FitSettings(method="sigmoid", regularization=1, tolerance=1e-9)
    fit = fit_calibrator(make_batch(probabilities, outcomes), settings)
    reference = LogisticRegression(C=1, solver="lbfgs", tol=1e-9, max_iter=settings.max_iterations)
    reference.fit(np.asarray(probabilities).reshape(-1, 1), outcomes)
    assert predict_confidence(fit, [0, 0.5, 1]) == pytest.approx(reference.predict_proba([[0], [0.5], [1]])[:, 1])


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("p,y,expected", [([0.5, 0.5], [0, 1], 0.5), ([0.1, 0.9], [1, 1], 1)])
def test_degenerate_fit_has_explicit_constant_state(make_batch, method, p, y, expected):
    fit = fit_calibrator(make_batch(p, y), FitSettings(method=method))
    assert fit.state.kind == "constant"
    assert predict_confidence(fit, [0, 0.5, 1]) == (expected, expected, expected)


def test_anticorrelated_sigmoid_uses_constrained_constant_optimum(make_batch):
    fit = fit_calibrator(make_batch([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), FitSettings(method="sigmoid"))
    assert predict_confidence(fit, [0, 1]) == (0.5, 0.5)
    assert fit.state.kind == "constant"


@pytest.mark.parametrize("role", ["validation", "test", "analysis"])
def test_no_implicit_refit_on_other_roles(make_batch, role):
    with pytest.raises(ValueError, match="calibration"):
        fit_calibrator(make_batch([0.5], [1], role=role), FitSettings())


def test_fitting_empty_data_is_unavailable_not_fabricated(make_batch):
    with pytest.raises(ValueError, match="valid"):
        fit_calibrator(make_batch([], []), FitSettings())


@pytest.mark.parametrize("change", [{"version": 77}, {"unknown": True}, {"artifact_id": "wrong"}])
def test_artifact_rejects_tampering_and_unrecognized_version(make_batch, change):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    payload = fit.model_dump(mode="json")
    payload.update(change)
    with pytest.raises(ValueError):
        FitArtifact.model_validate_json(json.dumps(payload))


def test_artifact_rejects_nonfinite_or_nonmonotone_state(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings(method="isotonic"))
    payload = fit.model_dump(mode="json")
    payload["state"]["y_thresholds"] = [1, 0]
    with pytest.raises(ValueError):
        FitArtifact.model_validate_json(json.dumps(payload))
    payload["state"]["y_thresholds"] = [0, float("inf")]
    with pytest.raises(ValueError):
        FitArtifact.model_validate_json(json.dumps(payload))


def test_scope_contract_mismatch_needs_recorded_override(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    test = make_batch([0.6], [1], role="test", prefix="test")
    changed = test.model_copy(update={"scope": CalibrationScope(judge_id="other", model_id="other", scope_id="other")})
    with pytest.raises(ValueError, match="compatib"):
        apply_calibrator(fit, changed)
    applied = apply_calibrator(fit, changed, override_reason="explicit constructed domain transfer")
    assert applied.overrides[-1].reason == "explicit constructed domain transfer"
    assert "scope" in applied.overrides[-1].mismatches


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_prediction_does_not_clip_invalid_inputs_into_validity(make_batch, probability):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    with pytest.raises(ValueError):
        predict_confidence(fit, [probability])
