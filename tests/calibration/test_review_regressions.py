"""Public API regressions from independent review and the artifact-boundary ruling."""

import math

import pytest

from openjudge.analyzer.validation import CalibrationAnalyzer
from openjudge.calibration import (
    BootstrapSettings,
    CalibrationContract,
    DecisionAdapter,
    FitSettings,
    SelectionSettings,
    WorkflowSettings,
    apply_calibrator,
    evaluate_policy,
    extract_observations,
    fit_calibrator,
    group_bootstrap,
    predict_confidence,
    reliability_metrics,
    select_policy,
)
from openjudge.calibration.artifacts import FitState
from openjudge.graders.schema import GraderError, GraderScore


@pytest.mark.parametrize("clip", [1e-20, 5e-324])
@pytest.mark.parametrize("p,y", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_small_clip_preserves_finite_loss_for_both_endpoint_outcomes(make_batch, clip, p, y):
    batch = make_batch([p], [y])
    result = reliability_metrics(batch, log_loss_clip=clip)
    expected = -math.log1p(-clip) if p == y else -math.log(clip)
    assert result.log_loss == expected
    assert math.isfinite(result.log_loss)
    assert result.log_loss > 0


def test_native_analyzer_and_bootstrap_share_endpoint_safe_loss(contract, scope):
    dataset = [{"id": "one", "group": "one", "label": 1}]
    results = [GraderScore(name="j", score=1, reason="", metadata={"confidence": 1})]
    analyzer = CalibrationAnalyzer(contract, scope, log_loss_clip=1e-20)
    assert analyzer.analyze(dataset, results).reliability.log_loss == -math.log1p(-1e-20)
    batch = extract_observations(dataset, results, contract, scope)
    uncertainty = group_bootstrap(batch, BootstrapSettings(resamples=5), log_loss_clip=1e-20)
    assert uncertainty.intervals["log_loss"].lower == -math.log1p(-1e-20)
    assert uncertainty.intervals["log_loss"].effective_resamples == 5


@pytest.fixture
def artifacts(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9, 0.3], [1, 0], role="validation", prefix="validation")
    return fit, select_policy(validation, fit, SelectionSettings(risk_target=0))


@pytest.mark.parametrize("flag", ["true", 1, [], {}])
def test_adapter_and_offline_extraction_reject_same_malformed_exclusion(artifacts, contract, scope, flag):
    contract = contract.model_copy(update={"exclude_path": "omit"})
    # Both supplied artifacts declare this actual extraction contract. The
    # compatibility override is explicit; malformed row input remains an error.
    adapter = DecisionAdapter(*artifacts, contract, scope, override_reason="exercise explicit exclusion input")
    native = GraderScore(name="j", score=1, reason="", metadata={"confidence": 1})
    sample = {"id": "one", "group": "one", "label": 1, "omit": flag}
    with pytest.raises(ValueError, match="boolean"):
        extract_observations([sample], [native], contract, scope)
    with pytest.raises(ValueError, match="boolean"):
        adapter.decide(native, sample)


@pytest.mark.parametrize("flag,expected", [(None, "accepted"), (False, "accepted"), (True, "unavailable")])
def test_adapter_valid_exclusion_flags_and_provider_precedence(artifacts, contract, scope, flag, expected):
    contract = contract.model_copy(update={"exclude_path": "omit"})
    adapter = DecisionAdapter(*artifacts, contract, scope, override_reason="exercise explicit exclusion input")
    native = GraderScore(name="j", score=1, reason="", metadata={"confidence": 1})
    assert adapter.decide(native, {"omit": flag}).status == expected
    provider = adapter.decide(GraderError(name="j", reason="", error="timeout"), {"omit": "malformed"})
    assert provider.status == "unavailable"
    assert provider.reason == "provider_error"


@pytest.mark.parametrize("boundary", ["predict", "apply", "select", "adapter"])
def test_copied_fit_state_with_stale_id_fails_before_public_inference(artifacts, make_batch, contract, scope, boundary):
    fit, policy = artifacts
    changed = fit.model_copy(update={"state": FitState(kind="constant", constant=0)})
    test = make_batch([0.9], [1], role="test", prefix="test")
    validation = make_batch([0.9], [1], role="validation", prefix="validation-new")
    with pytest.raises(ValueError, match="checksum"):
        if boundary == "predict":
            predict_confidence(changed, [0.9])
        elif boundary == "apply":
            apply_calibrator(changed, test)
        elif boundary == "select":
            select_policy(validation, changed)
        else:
            DecisionAdapter(changed, policy, contract, scope)


@pytest.mark.parametrize("boundary", ["evaluate", "adapter"])
def test_copied_policy_with_stale_id_fails_before_public_inference(artifacts, make_batch, contract, scope, boundary):
    fit, policy = artifacts
    changed = policy.model_copy(update={"threshold": 0.0})
    with pytest.raises(ValueError, match="checksum"):
        if boundary == "evaluate":
            evaluate_policy(
                make_batch([0.9], [1], role="test", prefix="test"),
                fit,
                changed,
                WorkflowSettings(bootstrap=BootstrapSettings(resamples=0)),
            )
        else:
            DecisionAdapter(fit, changed, contract, scope)


def test_adapter_owns_snapshot_and_does_not_hash_source_ids_per_decision(artifacts, contract, scope, monkeypatch):
    fit, policy = artifacts
    adapter = DecisionAdapter(fit, policy, contract, scope)
    fit.domain_counts["external mutation"] = 1
    assert "external mutation" not in adapter.artifact.domain_counts
    exposed = adapter.artifact
    exposed.domain_counts["exposed mutation"] = 1
    assert "exposed mutation" not in adapter.artifact.domain_counts

    def unexpected_rehash(_value):
        raise AssertionError("adapter must validate artifacts once, not per decision")

    monkeypatch.setattr("openjudge.calibration.artifacts.canonical_digest", unexpected_rehash)
    native = GraderScore(name="j", score=1, reason="", metadata={"confidence": 1})
    assert all(envelope.status == "accepted" for envelope in adapter.decide_many([native] * 100))
