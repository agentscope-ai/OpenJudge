"""Prove label separation through real runner results and safe cached replay."""

import json

import pytest

from openjudge.calibration.adapter import DecisionAdapter
from openjudge.calibration.cache import CachedPartition
from openjudge.calibration.contracts import CalibrationContract
from openjudge.calibration.fitting import FitSettings
from openjudge.calibration.metrics import BootstrapSettings
from openjudge.calibration.selective import SelectionSettings
from openjudge.calibration.workflow import WorkflowSettings, run_calibration_workflow
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.runner.grading_runner import GradingRunner


def partition(prefix, role, labels=(0, 1, 1)):
    data = [
        {
            "id": f"{prefix}-{i}",
            "group": f"{prefix}-g-{i}",
            "label": label,
            "query": "private prompt",
            "prediction": 1,
            "confidence": p,
        }
        for i, (label, p) in enumerate(zip(labels, (0.2, 0.8, 0.9)))
    ]
    results = [
        GraderScore(name="j", score=1, reason="private reasoning", metadata={"confidence": row["confidence"]})
        for row in data
    ]
    return CachedPartition(data, results, role=role)


def settings():
    return WorkflowSettings(
        fit=FitSettings(method="isotonic"),
        selection=SelectionSettings(risk_target=0.1),
        bootstrap=BootstrapSettings(resamples=10, seed=7),
    )


@pytest.mark.asyncio
async def test_real_function_grader_runner_cache_and_report(tmp_path, contract, scope):
    def evaluate(prediction, confidence):
        return GraderScore(name="j", score=prediction, reason="native", metadata={"confidence": confidence})

    grader = FunctionGrader(func=evaluate, name="j")
    runner = GradingRunner(grader_configs={"j": (grader, {"prediction": "prediction", "confidence": "confidence"})})
    inputs = []
    for prefix, role in (("fit", "calibration"), ("val", "validation"), ("test", "test")):
        source = partition(prefix, role)
        results = await runner.arun(list(source.dataset))
        cached = CachedPartition.from_runner(source.dataset, results, "j", role=role)
        path = tmp_path / f"{role}.json"
        cached.save(path)
        loaded = CachedPartition.load(path)
        assert [row["id"] for row in loaded.dataset] == [row["id"] for row in source.dataset]
        assert [result.score for result in loaded.grader_results] == [1, 1, 1]
        inputs.append(loaded)
    report = run_calibration_workflow(*inputs, contract, scope, settings())
    assert report.fit_artifact.source.role == "calibration"
    assert report.policy.validation_source.role == "validation"
    assert report.test.calibrated.reliability.brier == 0
    assert report.test.selected.accepted_count == 2
    assert report.test.selected.risk == 0
    assert report.test.always_accept.accepted_count == 3
    report.save(tmp_path / "report.json")
    text = (tmp_path / "report.json").read_text()
    assert "private prompt" not in text
    assert "private reasoning" not in text
    assert '"label":' not in text
    adapter = DecisionAdapter(report.fit_artifact, report.policy, contract, scope)
    restored = type(report).load(tmp_path / "report.json")
    replay = DecisionAdapter(restored.fit_artifact, restored.policy, contract, scope)
    assert [d.status for d in adapter.decide_many(inputs[2].grader_results)] == ["abstained", "accepted", "accepted"]
    assert [d.status for d in replay.decide_many(inputs[2].grader_results)] == ["abstained", "accepted", "accepted"]


def test_test_labels_cannot_affect_fit_or_threshold(contract, scope):
    fit, validation, test = partition("fit", "calibration"), partition("val", "validation"), partition("test", "test")
    first = run_calibration_workflow(fit, validation, test, contract, scope, settings())
    changed = run_calibration_workflow(
        fit, validation, partition("test", "test", (1, 0, 0)), contract, scope, settings()
    )
    assert first.fit_artifact.artifact_id == changed.fit_artifact.artifact_id
    assert first.policy.artifact_id == changed.policy.artifact_id
    assert first.test.selected.risk != changed.test.selected.risk
    changed_validation = run_calibration_workflow(
        fit, partition("val", "validation", (1, 0, 0)), test, contract, scope, settings()
    )
    assert first.fit_artifact.artifact_id == changed_validation.fit_artifact.artifact_id
    assert first.policy.artifact_id != changed_validation.policy.artifact_id


def test_three_way_leakage_audit_includes_excluded_rows(contract, scope):
    fit = partition("fit", "calibration")
    validation = partition("val", "validation")
    dataset = list(partition("test", "test").dataset)
    dataset[0] = {**dataset[0], "group": "fit-g-0", "omit": True}
    test = CachedPartition(dataset, partition("test", "test").grader_results, role="test")
    selected_contract = contract.model_copy(update={"exclude_path": "omit"})
    with pytest.raises(ValueError, match="group.*overlap"):
        run_calibration_workflow(fit, validation, test, selected_contract, scope, settings())


@pytest.mark.parametrize("change", [{"version": 2}, {"version": True}, {"unknown": 1}])
def test_cache_rejects_unknown_schema(tmp_path, change):
    path = tmp_path / "cache.json"
    partition("fit", "calibration").save(path)
    payload = json.loads(path.read_text())
    payload.update(change)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        CachedPartition.load(path)


def test_cache_rejects_nonfinite_duplicate_keys_and_size(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text('{"version":1,"version":2}')
    with pytest.raises(ValueError, match="duplicate"):
        CachedPartition.load(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        CachedPartition.load(path)
    partition("fit", "calibration").save(path)
    with pytest.raises(ValueError, match="exceeds"):
        CachedPartition.load(path, max_bytes=10)


def test_cache_and_workflow_preserve_provider_failures(tmp_path, contract, scope):
    source = partition("test", "test")
    results = list(source.grader_results)
    results[0] = GraderError(name="j", reason="", error="timeout")
    source = CachedPartition(source.dataset, results, role="test")
    source.save(tmp_path / "errors.json")
    loaded = CachedPartition.load(tmp_path / "errors.json")
    assert isinstance(loaded.grader_results[0], GraderError)
    report = run_calibration_workflow(
        partition("fit", "calibration"), partition("val", "validation"), loaded, contract, scope, settings()
    )
    assert report.test.selected.provider_error_count == 1
    assert report.test.selected.unconditional_error_count == 1
    assert report.test.selected.coverage == pytest.approx(2 / 3)
