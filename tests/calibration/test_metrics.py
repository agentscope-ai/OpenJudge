"""Numerical oracles independent of metric implementations."""

import math

import numpy as np
import pytest

from openjudge.calibration.metrics import (
    BootstrapSettings,
    group_bootstrap,
    reliability_metrics,
)


def test_hand_computed_reliability_and_endpoint_bins(make_batch):
    result = reliability_metrics(make_batch([0, 0.5, 1], [0, 1, 1]), bin_edges=(0, 0.5, 1), log_loss_clip=0.01)
    assert [b.count for b in result.bins] == [1, 2]
    assert result.brier == pytest.approx(1 / 12)
    assert result.ece == pytest.approx(1 / 6)
    assert result.mce == pytest.approx(0.25)
    assert result.log_loss == pytest.approx((-2 * math.log(0.99) - math.log(0.5)) / 3)
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.log_loss_clip == 0.01
    assert result.sample_count == 3


def test_empty_bins_and_empty_batch_use_null(make_batch):
    result = reliability_metrics(make_batch([1], [1]), bin_edges=(0, 0.25, 0.5, 1))
    assert result.bins[0].mean_confidence is None
    assert result.bins[0].accuracy is None
    empty = reliability_metrics(make_batch([], []))
    assert empty.ece is None
    assert empty.brier is None
    assert empty.accuracy is None
    assert empty.sample_count == 0


@pytest.mark.parametrize("edges", [(0, 0.5), (0.1, 1), (0, 0.8, 0.7, 1), (0, 0, 1), (0, float("nan"), 1)])
def test_invalid_bins_fail(make_batch, edges):
    with pytest.raises(ValueError):
        reliability_metrics(make_batch([0.5], [1]), bin_edges=edges)


@pytest.mark.parametrize("clip", [0, 0.5, -0.1, float("nan")])
def test_invalid_loss_clip_fails(make_batch, clip):
    with pytest.raises(ValueError):
        reliability_metrics(make_batch([0.5], [1]), log_loss_clip=clip)


def test_group_bootstrap_resamples_whole_groups_and_preserves_rng(make_batch):
    batch = make_batch([0, 0, 1, 1], [0, 0, 1, 1], groups=["a", "a", "b", "b"])
    settings = BootstrapSettings(resamples=40, seed=12, confidence_level=0.9)
    np.random.seed(921)
    expected_global_next = np.random.random()
    np.random.seed(921)
    result = group_bootstrap(batch, settings)
    assert np.random.random() == expected_global_next
    assert result == group_bootstrap(batch, settings)
    assert result.group_count == 2
    assert result.intervals["brier"].lower == 0
    assert result.intervals["brier"].upper == 0
    assert result.intervals["accuracy"].lower == 0
    assert result.intervals["accuracy"].upper == 1
    assert "descriptive" in result.interpretation


def test_domain_metrics_retain_counts_without_row_payloads(make_batch):
    result = reliability_metrics(make_batch([0.2, 0.8], [0, 1], domains=["easy", "hard"]))
    assert result.domains["easy"].sample_count == 1
    assert result.domains["hard"].brier == pytest.approx(0.04)
    assert "observations" not in result.model_dump()
