"""Contracts must prevent accidental score substitution and split leakage."""

import math

import pytest

from openjudge.calibration.contracts import CalibrationContract
from openjudge.calibration.extraction import extract_observations, vote_fraction
from openjudge.calibration.splits import audit_partitions, partition_provenance
from openjudge.graders.schema import GraderError, GraderScore


def score(p=0.8, prediction=1):
    return GraderScore(name="judge", score=prediction, reason="", metadata={"confidence": p})


def test_extracts_detached_ordered_correctness_with_provider_failure(contract, scope):
    data = [{"id": "a", "group": "one", "label": 1}, {"id": "b", "group": "two", "label": 0}]
    results = [score(), GraderError(name="judge", reason="", error="timeout")]
    batch = extract_observations(data, results, contract, scope, role="analysis")
    assert [(o.sample_id, o.status, o.correctness) for o in batch.observations] == [
        ("a", "valid", 1),
        ("b", "provider_error", None),
    ]
    results[0].metadata["confidence"] = 0.2
    data[0]["id"] = "changed"
    assert batch.observations[0].probability == 0.8
    assert batch.observations[0].sample_id == "a"
    assert batch.counts["provider_error"] == 1
    with pytest.raises((AttributeError, ValueError)):
        batch.observations[0].probability = 0.4


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf, "0.8", True])
def test_invalid_probability_rejected_or_explicitly_counted(contract, scope, probability):
    data = [{"id": "a", "group": "one", "label": 1}]
    with pytest.raises(ValueError, match="invalid_confidence"):
        extract_observations(data, [score(probability)], contract, scope)
    batch = extract_observations(data, [score(probability)], contract, scope, invalid_policy="exclude")
    assert batch.counts["invalid_confidence"] == 1
    assert batch.observations[0].probability is None


@pytest.mark.parametrize(
    "sample,result,status",
    [
        ({"id": "a", "group": "g", "label": 1}, GraderScore(name="j", score=0.9, reason=""), "invalid_prediction"),
        ({"id": "a", "group": "g", "label": 1}, score(None), "missing_confidence"),
        ({"id": "a", "group": "g"}, score(), "missing_label"),
        ({"id": "a", "group": "g", "label": 3}, score(), "invalid_label"),
    ],
)
def test_exclusion_statuses_do_not_guess_targets_or_confidence(contract, scope, sample, result, status):
    with pytest.raises(ValueError, match=status):
        extract_observations([sample], [result], contract, scope)
    assert extract_observations([sample], [result], contract, scope, invalid_policy="exclude").counts[status] == 1


def test_missing_confidence_never_uses_valid_decision_score(contract, scope):
    result = GraderScore(name="j", score=1, reason="")
    with pytest.raises(ValueError, match="missing_confidence"):
        extract_observations([{"id": "a", "group": "g", "label": 1}], [result], contract, scope)


def test_alignment_and_identity_fail_even_when_excluding_invalid_rows(contract, scope):
    data = [{"id": "a", "group": "g", "label": 1}]
    with pytest.raises(ValueError, match="same length"):
        extract_observations(data, [], contract, scope)
    with pytest.raises(ValueError, match="duplicate"):
        extract_observations(data * 2, [score(), score()], contract, scope, invalid_policy="exclude")
    with pytest.raises(ValueError, match="group"):
        extract_observations([{"id": "a", "label": 1}], [score()], contract, scope, invalid_policy="exclude")


def test_user_exclusion_remains_visible_and_audited(scope):
    contract = CalibrationContract(target_id="correctness", confidence_path="metadata.confidence", exclude_path="omit")
    data = [{"id": "a", "group": "same", "omit": True}, {"id": "b", "group": "same", "label": 1}]
    batch = extract_observations(data, [score(None), score()], contract, scope)
    assert batch.counts["user_excluded"] == 1
    assert partition_provenance(batch).group_ids == ("same",)
    other = extract_observations(
        [{"id": "c", "group": "same", "label": 1}], [score()], contract, scope, role="validation"
    )
    with pytest.raises(ValueError, match="group.*overlap"):
        audit_partitions(batch, other)


def test_manifests_detect_distinct_ids_from_same_task(make_batch):
    fit = make_batch([0.2, 0.9], [0, 1])
    validation = make_batch([0.7], [1], "validation", "validation", groups=["fit-task-0"])
    with pytest.raises(ValueError, match="group.*overlap"):
        audit_partitions(fit, validation)
    test = make_batch([0.7], [1], "test", "test")
    validation = make_batch([0.7], [1], "validation", "validation")
    manifest = audit_partitions(fit, validation, test)
    assert len(manifest.partitions) == 3
    assert manifest.partitions[0].sample_count == 2
    assert partition_provenance(fit).fingerprint == partition_provenance(fit).fingerprint


def test_explicit_callback_and_vote_interpretation(scope):
    contract = CalibrationContract(
        target_id="correctness",
        confidence_path=None,
        extractor_id="vote_fraction_v1",
        confidence_interpretation="fraction of successful repeated votes agreeing with selected decision",
    )
    result = score()
    result.metadata = {"original_results": [score(prediction=1), score(prediction=1), score(prediction=0)]}
    data = [{"id": "a", "group": "g", "label": 1}]
    batch = extract_observations(data, [result], contract, scope, confidence_extractor=vote_fraction)
    assert batch.observations[0].probability == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="extractor"):
        extract_observations(data, [result], contract, scope)
    with pytest.raises(ValueError):
        CalibrationContract(target_id="correctness", confidence_path=None)


def test_explicit_dataset_confidence_source(scope):
    contract = CalibrationContract(target_id="correctness", confidence_path="confidence", confidence_source="dataset")
    batch = extract_observations(
        [{"id": "a", "group": "g", "label": 1, "confidence": 0.6}], [score(0.1)], contract, scope
    )
    assert batch.observations[0].probability == 0.6
