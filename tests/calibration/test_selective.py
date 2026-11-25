"""Tie handling and threshold selection must not manufacture success."""

import math

import pytest

from openjudge.calibration.extraction import extract_observations
from openjudge.calibration.fitting import FitSettings, fit_calibrator
from openjudge.calibration.selective import (
    SelectionSettings,
    risk_coverage_curve,
    select_policy,
    selective_metrics,
)
from openjudge.graders.schema import GraderError, GraderScore


def test_tie_group_aurc_hand_oracle_and_order_invariance(make_batch):
    batch = make_batch([0.9, 0.9, 0.2], [1, 0, 1], role="test", prefix="test")
    result = risk_coverage_curve(batch)
    assert [p.accepted_count for p in result.points] == [0, 2, 3]
    assert result.points[0].risk is None
    assert result.points[1].risk == 0.5
    assert result.aurc == pytest.approx(4 / 9)
    reversed_tie = make_batch([0.9, 0.9, 0.2], [0, 1, 1], role="test", prefix="test")
    assert risk_coverage_curve(reversed_tie).aurc == result.aurc


def test_provider_failure_is_not_successful_abstention(contract, scope):
    batch = extract_observations(
        [{"id": "a", "group": "a", "label": 1}, {"id": "b", "group": "b", "label": 0}],
        [
            GraderScore(name="j", score=1, reason="", metadata={"confidence": 0.9}),
            GraderError(name="j", reason="", error="timeout"),
        ],
        contract,
        scope,
        role="test",
    )
    result = selective_metrics(batch, threshold=None)
    assert result.accepted_count == 0
    assert result.abstained_count == 1
    assert result.provider_error_count == 1
    assert result.risk is None
    assert result.accepted_accuracy is None
    assert result.unconditional_error_count == 1
    assert result.unconditional_error_rate == 0.5
    assert result.coverage == 0
    curve = risk_coverage_curve(batch)
    assert curve.points[-1].coverage == 0.5
    assert curve.points[-1].evaluable_coverage == 1


def test_validation_selects_max_coverage_feasible_tie_group(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9, 0.9, 0.4], [1, 1, 0], role="validation", prefix="validation")
    policy = select_policy(validation, fit, SelectionSettings(risk_target=0, min_accepted=2))
    assert policy.threshold == 0.9
    assert policy.selection.accepted_count == 2
    assert policy.selection.risk == 0
    assert policy.criterion == "empirical"
    assert policy.guarantee is None
    assert policy.validation_source.role == "validation"
    assert policy.fit_artifact_id == fit.artifact_id


def test_empty_feasible_set_emits_abstain_all(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9, 0.4], [0, 0], role="validation", prefix="validation")
    policy = select_policy(validation, fit, SelectionSettings(risk_target=0))
    assert policy.threshold is None
    assert policy.selection.risk is None
    assert policy.reason == "no_feasible_threshold"
    assert policy.selection.accepted_count == 0


def test_fixed_grid_hoeffding_adjusts_all_declared_thresholds(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9] * 100, [1] * 100, role="validation", prefix="validation")
    settings = SelectionSettings(
        risk_target=0.5, criterion="hoeffding", candidates=(0.5, 0.8), alpha=0.05, assume_iid_observations=True
    )
    policy = select_policy(validation, fit, settings)
    assert policy.threshold == 0.5
    assert policy.selection_upper_risk == pytest.approx(math.sqrt(math.log(2 / 0.05) / 200))
    assert policy.candidate_count == 2
    assert "independent" in policy.guarantee


def test_bound_requires_fixed_grid_and_explicit_independence(make_batch):
    with pytest.raises(ValueError, match="fixed"):
        SelectionSettings(criterion="hoeffding", assume_iid_observations=True)
    with pytest.raises(ValueError, match="independent"):
        SelectionSettings(criterion="hoeffding", candidates=(0.5,))
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    validation = make_batch([0.9, 0.9], [1, 1], role="validation", prefix="validation", groups=["v", "v"])
    with pytest.raises(ValueError, match="group"):
        select_policy(
            validation, fit, SelectionSettings(criterion="hoeffding", candidates=(0.5,), assume_iid_observations=True)
        )


def test_fit_validation_overlap_fails_and_test_cannot_select(make_batch):
    fit = fit_calibrator(make_batch([0.2, 0.8], [0, 1]), FitSettings())
    with pytest.raises(ValueError, match="overlap"):
        select_policy(make_batch([0.2], [1], role="validation"), fit, SelectionSettings())
    with pytest.raises(ValueError, match="validation"):
        select_policy(make_batch([0.2], [1], role="test", prefix="test"), fit, SelectionSettings())
