"""Real native results remain unchanged through optional decisions."""

import pytest

from openjudge.analyzer.validation import CalibrationAnalyzer, SelectivePredictionAnalyzer
from openjudge.analyzer.validation.base_validation_analyzer import BaseValidationAnalyzer
from openjudge.calibration.adapter import DecisionAdapter
from openjudge.calibration.artifacts import FitArtifact, PolicyArtifact
from openjudge.calibration.fitting import FitSettings, fit_calibrator
from openjudge.calibration.selective import SelectionSettings, select_policy
from openjudge.graders.schema import GraderError, GraderScore


def test_native_analyzers_use_same_oracles_and_coverage(contract, scope):
    dataset = [{"id": "a", "group": "a", "label": 1}, {"id": "b", "group": "b", "label": 0}]
    results = [GraderScore(name="j", score=1, reason="", metadata={"confidence": 0.5}),
               GraderError(name="j", reason="", error="timeout")]
    analyzer = CalibrationAnalyzer(contract, scope)
    assert isinstance(analyzer, BaseValidationAnalyzer)
    result = analyzer.analyze(dataset, results)
    assert result.reliability.brier == 0.25
    assert result.metadata["error_results"] == 1
    assert result.metadata["valid_results"] == 1
    selective = SelectivePredictionAnalyzer(contract, scope, threshold=None).analyze(dataset, results)
    assert selective.operating.risk is None
    assert selective.operating.provider_error_count == 1
    assert selective.metadata["coverage_rate"] == 0.5
    with pytest.raises(ValueError, match="same length"):
        analyzer.analyze(dataset, [])
    assert analyzer.analyze([], []).reliability.brier is None


def test_adapter_preserves_native_result_and_reloads_exact_decisions(make_batch, contract, scope):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9, 0.3], [1, 0], role="validation", prefix="validation")
    policy = select_policy(validation, fit, SelectionSettings(risk_target=0))
    adapter = DecisionAdapter(fit, policy, contract, scope)
    restored = DecisionAdapter(
        FitArtifact.model_validate_json(fit.model_dump_json()),
        PolicyArtifact.model_validate_json(policy.model_dump_json()), contract, scope
    )
    for p, expected in [(0.95, "accepted"), (0.2, "abstained")]:
        native = GraderScore(name="j", score=1, reason="unchanged", metadata={"confidence": p})
        before = native.model_dump()
        decision = adapter.decide(native)
        assert decision.status == expected
        assert decision.original_result is native
        assert decision.fit_artifact_id == fit.artifact_id
        assert decision.policy_id == policy.artifact_id
        assert restored.decide(native).status == decision.status
        assert native.model_dump() == before


def test_adapter_missing_confidence_or_provider_error_unavailable(make_batch, contract, scope):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    policy = select_policy(
        make_batch([0.9], [1], role="validation", prefix="validation"), fit, SelectionSettings()
    )
    adapter = DecisionAdapter(fit, policy, contract, scope)
    error = adapter.decide(GraderError(name="j", reason="", error="timeout"))
    missing = adapter.decide(GraderScore(name="j", score=1, reason=""))
    assert error.status == "unavailable"
    assert error.reason == "provider_error"
    assert missing.status == "unavailable"
    assert missing.reason == "missing_confidence"
