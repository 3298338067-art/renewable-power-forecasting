import numpy as np
import pytest

from renewable_forecasting.evaluation_analysis import (
    aggregate_prediction_diagnostics,
    moving_block_origin_indices,
    paired_moving_block_bootstrap,
    training_derived_masks,
)


DAY_NS = 86_400_000_000_000


def test_training_derived_masks_use_only_training_thresholds():
    train = {
        "history_power": np.zeros((2, 2, 1), dtype=np.float32),
        "target_power": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }
    evaluated = {
        "history_power": np.array([[[0.0], [1.0]]], dtype=np.float32),
        "target_power": np.array([[2.0, 3.0]], dtype=np.float32),
    }

    result = training_derived_masks(
        train,
        evaluated,
        high_power_quantile=0.5,
        ramp_quantile=0.5,
    )

    assert result["thresholds"]["high_power"] == pytest.approx(2.5)
    assert result["thresholds"]["ramp_magnitude"] == pytest.approx(1.0)
    np.testing.assert_array_equal(result["masks"]["high_power"], [[False, True]])
    np.testing.assert_array_equal(result["masks"]["ramp"], [[True, True]])
    assert result["thresholds"]["derived_from"] == "training_target_power"


def test_moving_blocks_keep_consecutive_dates_inside_each_block():
    origins = np.repeat(np.arange(8, dtype=np.int64) * DAY_NS, 3)
    zones = np.tile(np.array([1, 2, 3]), 8)

    sampled = moving_block_origin_indices(
        origins,
        zones,
        block_length_days=3,
        n_resamples=5,
        random_seed=17,
    )

    assert sampled.shape == (5, 8)
    for replicate in sampled:
        np.testing.assert_array_equal(np.diff(replicate[:3]), [1, 1])
        np.testing.assert_array_equal(np.diff(replicate[3:6]), [1, 1])


def test_paired_bootstrap_recomputes_overall_rmse_before_difference():
    dates = 4
    zones_per_date = 3
    horizon = 2
    origins = np.repeat(np.arange(dates, dtype=np.int64) * DAY_NS, zones_per_date)
    zones = np.tile(np.array([1, 2, 3]), dates)
    actual = np.zeros((dates * zones_per_date, horizon), dtype=np.float32)
    candidate_one_seed = np.zeros_like(actual)
    candidate_one_seed[:zones_per_date] = 2.0
    reference_one_seed = np.full_like(actual, 0.5)
    candidate = np.stack([candidate_one_seed, candidate_one_seed])
    reference = np.stack([reference_one_seed, reference_one_seed])

    result = paired_moving_block_bootstrap(
        actual,
        candidate,
        reference,
        origins,
        zones,
        block_length_days=2,
        n_resamples=100,
        random_seed=9,
        metric="rmse",
    )

    assert result["point_estimate"] == pytest.approx(0.5)
    assert result["difference"] == "candidate_minus_reference"
    assert result["negative_favors"] == "candidate"
    assert result["confidence_interval_95"][0] <= result["confidence_interval_95"][1]
    assert result["n_resamples"] == 100


def test_paired_bootstrap_rejects_incomplete_date_zone_groups():
    origins = np.array([0, 0, 0, DAY_NS, DAY_NS], dtype=np.int64)
    zones = np.array([1, 2, 3, 1, 2])
    actual = np.zeros((5, 2), dtype=np.float32)
    predictions = np.zeros((2, 5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same zone set"):
        paired_moving_block_bootstrap(
            actual,
            predictions,
            predictions,
            origins,
            zones,
            block_length_days=1,
            n_resamples=10,
            random_seed=1,
            metric="mae",
        )


def test_aggregate_diagnostics_reports_seed_mean_and_subgroups():
    actual = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    predictions = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    masks = {
        "high_power": np.array([[False, True], [False, True]]),
        "ramp": np.array([[False, True], [False, True]]),
    }

    result = aggregate_prediction_diagnostics(
        actual,
        predictions,
        np.array([1, 2]),
        masks,
    )

    assert result["seed_count"] == 2
    assert result["aggregate"]["overall"]["mae"]["mean"] == pytest.approx(0.25)
    assert result["aggregate"]["high_power"]["rmse"]["mean"] == pytest.approx(0.5)
    assert set(result["aggregate"]["by_zone"]) == {"1", "2"}
    assert len(result["aggregate"]["by_horizon"]) == 2
