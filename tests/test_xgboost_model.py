import numpy as np
import pytest

from renewable_forecasting.xgboost_model import (
    aggregate_seed_metrics,
    build_long_horizon_table,
    clip_normalized_power,
    reshape_long_predictions,
    select_best_candidate,
)


def make_split() -> dict[str, np.ndarray]:
    return {
        "history_power": np.arange(48, dtype=np.float32).reshape(2, 24, 1),
        "future_nwp_raw": (
            100 + np.arange(2 * 24 * 2, dtype=np.float32)
        ).reshape(2, 24, 2),
        "future_calendar": (
            300 + np.arange(2 * 24 * 4, dtype=np.float32)
        ).reshape(2, 24, 4),
        "target_power": (
            500 + np.arange(48, dtype=np.float32)
        ).reshape(2, 24),
        "zone_id": np.array([1, 3], dtype=np.int16),
    }


def test_build_long_horizon_table_aligns_features_and_targets():
    table = build_long_horizon_table(
        make_split(),
        nwp_feature_names=["cloud", "irradiance"],
        calendar_feature_names=[
            "utc_hour_sin",
            "utc_hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ],
        zone_categories=[1, 2, 3],
    )

    assert table.features.shape == (48, 34)
    assert table.targets.shape == (48,)
    assert len(table.feature_names) == 34
    assert len(set(table.feature_names)) == 34
    np.testing.assert_array_equal(table.features[0, :24], np.arange(24))
    np.testing.assert_array_equal(table.features[1, :24], np.arange(24))
    np.testing.assert_array_equal(table.features[24, :24], np.arange(24, 48))
    np.testing.assert_array_equal(table.features[0, 24:26], [100, 101])
    np.testing.assert_array_equal(table.features[1, 24:26], [102, 103])
    assert table.features[0, table.feature_names.index("horizon_hour")] == 1
    assert table.features[23, table.feature_names.index("horizon_hour")] == 24
    assert table.features[0, table.feature_names.index("zone_1")] == 1
    assert table.features[0, table.feature_names.index("zone_3")] == 0
    assert table.features[24, table.feature_names.index("zone_3")] == 1
    np.testing.assert_array_equal(table.targets, np.arange(500, 548))
    assert not any("target" in name for name in table.feature_names)


def test_build_long_horizon_table_rejects_unknown_zone():
    split = make_split()
    split["zone_id"][0] = 4

    with pytest.raises(ValueError, match="unknown zone"):
        build_long_horizon_table(
            split,
            nwp_feature_names=["cloud", "irradiance"],
            calendar_feature_names=["a", "b", "c", "d"],
            zone_categories=[1, 2, 3],
        )


def test_clip_normalized_power_applies_fixed_physical_bounds():
    predictions = np.array([[-0.2, 0.4, 1.2]], dtype=np.float32)

    clipped = clip_normalized_power(predictions, lower=0.0, upper=1.0)

    np.testing.assert_allclose(clipped, [[0.0, 0.4, 1.0]])


def test_reshape_long_predictions_restores_sample_horizon_order():
    flat = np.arange(48, dtype=np.float32)

    restored = reshape_long_predictions(flat, sample_count=2, horizon_hours=24)

    assert restored.shape == (2, 24)
    np.testing.assert_array_equal(restored[1], np.arange(24, 48))


def test_select_best_candidate_uses_validation_rmse_only():
    candidates = [
        {"name": "deeper", "validation_rmse": 0.18, "test_rmse": 0.01},
        {"name": "shallower", "validation_rmse": 0.12, "test_rmse": 0.40},
    ]

    selected = select_best_candidate(candidates)

    assert selected["name"] == "shallower"


def test_aggregate_seed_metrics_returns_mean_and_sample_std():
    runs = [
        {
            "validation": {
                "overall": {"mae": 0.10, "rmse": 0.20},
                "irradiance_active": {"mae": 0.15, "rmse": 0.25},
            },
            "test": {
                "overall": {"mae": 0.30, "rmse": 0.40},
                "irradiance_active": {"mae": 0.35, "rmse": 0.45},
            },
        },
        {
            "validation": {
                "overall": {"mae": 0.20, "rmse": 0.30},
                "irradiance_active": {"mae": 0.25, "rmse": 0.35},
            },
            "test": {
                "overall": {"mae": 0.40, "rmse": 0.50},
                "irradiance_active": {"mae": 0.45, "rmse": 0.55},
            },
        },
    ]

    aggregate = aggregate_seed_metrics(runs)

    assert aggregate["validation"]["overall"]["rmse"]["mean"] == pytest.approx(0.25)
    assert aggregate["validation"]["overall"]["rmse"]["std"] == pytest.approx(
        np.sqrt(0.005)
    )
    assert aggregate["test"]["irradiance_active"]["mae"]["mean"] == pytest.approx(
        0.40
    )
