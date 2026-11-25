"""Checksums do not replace semantic validation of hand-edited finite JSON."""

import json

import pytest

from openjudge.calibration import (
    DecisionAdapter,
    FitArtifact,
    FitSettings,
    PolicyArtifact,
    SelectionSettings,
    apply_calibrator,
    fit_calibrator,
    select_policy,
)
from openjudge.calibration.cache import CachedPartition
from openjudge.calibration.contracts import canonical_digest
from openjudge.graders.schema import GraderScore


def rehash(payload):
    return {
        **payload,
        "artifact_id": canonical_digest({key: value for key, value in payload.items() if key != "artifact_id"}),
    }


@pytest.fixture
def policy_pair(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    val = make_batch([0.9, 0.3], [1, 0], role="validation", prefix="validation")
    return fit, select_policy(val, fit, SelectionSettings(risk_target=0))


@pytest.mark.parametrize(
    "update",
    [
        {"selection_upper_risk": 0.5},
        {"criterion": "hoeffding", "assume_iid_observations": True},
        {"selection": {"accepted_count": 1, "error_count": 0, "risk": 0.0, "coverage": 1.0}},
    ],
)
def test_rehashed_inconsistent_policy_is_rejected(policy_pair, update):
    _, policy = policy_pair
    payload = policy.model_dump(mode="json")
    payload.update(update)
    with pytest.raises(ValueError):
        PolicyArtifact.model_validate_json(json.dumps(rehash(payload)))


def test_rehashed_nonmonotone_state_is_rejected(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings(method="isotonic"))
    payload = fit.model_dump(mode="json")
    payload["state"]["y_thresholds"] = [1, 0]
    with pytest.raises(ValueError, match="nonmonotone"):
        FitArtifact.model_validate_json(json.dumps(rehash(payload)))


def test_double_calibration_and_wrong_target_cannot_silently_apply(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    test = make_batch([0.6], [1], role="test", prefix="test")
    calibrated = apply_calibrator(fit, test)
    with pytest.raises(ValueError, match="already"):
        apply_calibrator(fit, calibrated)
    changed = test.model_copy(update={"contract": test.contract.model_copy(update={"target_id": "different-event"})})
    with pytest.raises(ValueError, match="compatib"):
        apply_calibrator(fit, changed)


def test_policy_cannot_bind_to_another_fit(policy_pair, make_batch, contract, scope):
    _, policy = policy_pair
    other_fit = fit_calibrator(make_batch([0.1, 0.8], [0, 1]), FitSettings())
    with pytest.raises(ValueError, match="different"):
        DecisionAdapter(other_fit, policy, contract, scope)


@pytest.mark.parametrize("bad_type", [[], {}])
def test_malformed_cache_tags_raise_validation_error(bad_type):
    result = GraderScore(name="j", score=1, reason="")
    payload = CachedPartition([{"id": "a"}], [result], role="test").to_payload()
    payload["records"][0]["result_type"] = bad_type
    payload["content_id"] = canonical_digest({key: value for key, value in payload.items() if key != "content_id"})
    with pytest.raises(ValueError):
        CachedPartition.from_payload(payload)
