import numpy as np
import pandas as pd
import pytest

from renewable_forecasting.baselines import (
    daily_persistence,
    seasonal_persistence,
)
from renewable_forecasting.metrics import (
    compute_error_metrics,
    summarize_forecast_errors,
)


def test_daily_persistence_aligns_each_target_with_previous_day():
    history = np.arange(48, dtype=np.float32).reshape(2, 24, 1)

    prediction = daily_persistence(history)

    assert prediction.shape == (2, 24)
    np.testing.assert_array_equal(prediction, history[:, :, 0])


def test_daily_persistence_rejects_invalid_history_shape():
    with pytest.raises(ValueError, match="samples, 24, 1"):
        daily_persistence(np.zeros((2, 24), dtype=np.float32))


def test_seasonal_persistence_uses_exact_zone_and_timestamp():
    panel = pd.DataFrame(
        {
            "zone_id": [1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                ]
            ),
            "power": [0.1, 0.2, 0.7, 0.8],
        }
    )
    targets = np.stack(
        [
            pd.to_datetime(["2020-01-08 01:00", "2020-01-08 02:00"]).asi8,
            pd.to_datetime(["2020-01-08 01:00", "2020-01-08 02:00"]).asi8,
        ]
    )

    prediction = seasonal_persistence(
        panel,
        zone_ids=np.array([1, 2], dtype=np.int16),
        target_timestamps_utc=targets,
    )

    np.testing.assert_allclose(
        prediction,
        np.array([[0.1, 0.2], [0.7, 0.8]], dtype=np.float32),
    )


def test_seasonal_persistence_rejects_missing_lagged_observation():
    panel = pd.DataFrame(
        {
            "zone_id": [1],
            "timestamp": pd.to_datetime(["2020-01-01 01:00"]),
            "power": [0.1],
        }
    )
    targets = pd.to_datetime(
        ["2020-01-08 01:00", "2020-01-08 02:00"]
    ).asi8.reshape(1, 2)

    with pytest.raises(ValueError, match="missing exact 7-day"):
        seasonal_persistence(panel, np.array([1]), targets)


def test_compute_error_metrics_matches_known_values():
    actual = np.array([[0.0, 2.0], [2.0, 4.0]])
    predicted = np.array([[0.0, 0.0], [4.0, 4.0]])

    metrics = compute_error_metrics(actual, predicted)

    assert metrics["count"] == 4
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["nrmse_capacity_percent"] == pytest.approx(100 * np.sqrt(2.0))


def test_summarize_forecast_errors_reports_zones_and_horizons():
    actual = np.array([[0.0, 2.0], [2.0, 4.0]])
    predicted = np.array([[0.0, 0.0], [4.0, 4.0]])

    summary = summarize_forecast_errors(
        actual,
        predicted,
        zone_ids=np.array([1, 2]),
    )

    assert summary["overall"]["count"] == 4
    assert set(summary["by_zone"]) == {"1", "2"}
    assert summary["by_zone"]["1"]["mae"] == pytest.approx(1.0)
    assert summary["by_zone"]["2"]["mae"] == pytest.approx(1.0)
    assert [row["horizon_hour"] for row in summary["by_horizon"]] == [1, 2]
    assert summary["by_horizon"][0]["rmse"] == pytest.approx(np.sqrt(2.0))
    assert summary["by_horizon"][1]["rmse"] == pytest.approx(np.sqrt(2.0))


def test_summarize_forecast_errors_reports_irradiance_active_metrics():
    actual = np.array([[0.0, 1.0], [0.0, 0.5]])
    predicted = np.array([[0.2, 0.5], [0.1, 1.0]])
    irradiance_active = np.array([[False, True], [False, True]])

    summary = summarize_forecast_errors(
        actual,
        predicted,
        zone_ids=np.array([1, 2]),
        irradiance_active_mask=irradiance_active,
    )

    assert summary["irradiance_active"]["count"] == 2
    assert summary["irradiance_active"]["mae"] == pytest.approx(0.5)
    assert summary["irradiance_active"]["rmse"] == pytest.approx(0.5)
    assert summary["irradiance_active"]["nrmse_capacity_percent"] == pytest.approx(
        50.0
    )


def test_summarize_forecast_errors_rejects_mismatched_irradiance_mask():
    with pytest.raises(ValueError, match="irradiance-active mask"):
        summarize_forecast_errors(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            zone_ids=np.array([1, 2]),
            irradiance_active_mask=np.ones((2, 1), dtype=bool),
        )
