import json

import numpy as np
import pandas as pd
import pytest

import renewable_forecasting.preprocessing as preprocessing
from renewable_forecasting.preprocessing import prepare_hourly_panel


def make_valid_panel_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01 01:00", periods=48, freq="h")
    return pd.DataFrame(
        {
            "ZONEID": 1,
            "TIMESTAMP": timestamps.strftime("%Y%m%d %H:%M"),
            "POWER": [index / 100 for index in range(48)],
            "VAR169": list(range(1, 25)) + list(range(1, 25)),
        }
    )


def make_window_panel(days: int = 3, zones: tuple[int, ...] = (1,)) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01 01:00", periods=days * 24, freq="h")
    frames = []
    for zone_id in zones:
        frames.append(
            pd.DataFrame(
                {
                    "zone_id": zone_id,
                    "timestamp": timestamps,
                    "forecast_day": (timestamps - pd.Timedelta(hours=1)).normalize(),
                    "power": np.arange(days * 24, dtype=np.float32),
                    "var78": 100 + np.arange(days * 24, dtype=np.float32),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_prepare_hourly_panel_differences_within_forecast_day_and_clips_negatives():
    timestamps = pd.date_range("2020-01-01 01:00", periods=48, freq="h")
    accumulated = list(range(10, 34)) + list(range(4, 28))
    accumulated[1] = 9
    frame = pd.DataFrame(
        {
            "ZONEID": 1,
            "TIMESTAMP": timestamps.strftime("%Y%m%d %H:%M"),
            "POWER": [index / 100 for index in range(48)],
            "VAR169": accumulated,
        }
    )

    panel, audit = prepare_hourly_panel(frame, ["VAR169"])

    assert panel["var169_hourly"].iloc[[0, 1, 24, 25]].tolist() == [10, 0, 4, 1]
    assert panel["forecast_day"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert panel["forecast_day"].iloc[24] == pd.Timestamp("2020-01-02", tz="UTC")
    assert {"zone_id", "timestamp", "power"}.issubset(panel.columns)
    assert audit["VAR169"]["negative_count"] == 1
    assert audit["VAR169"]["minimum_raw_increment"] == -1.0


def test_prepare_hourly_panel_rejects_duplicate_zone_timestamps():
    frame = make_valid_panel_frame()
    frame.loc[1, "TIMESTAMP"] = frame.loc[0, "TIMESTAMP"]

    with pytest.raises(ValueError, match="duplicate zone-timestamp"):
        prepare_hourly_panel(frame, ["VAR169"])


def test_prepare_hourly_panel_rejects_non_hourly_gaps():
    frame = make_valid_panel_frame().drop(index=2)

    with pytest.raises(ValueError, match="hourly continuity"):
        prepare_hourly_panel(frame, ["VAR169"])


def test_prepare_hourly_panel_requires_24_rows_per_forecast_day():
    frame = make_valid_panel_frame().iloc[:-1]

    with pytest.raises(ValueError, match="24 hourly rows"):
        prepare_hourly_panel(frame, ["VAR169"])


def test_build_day_ahead_windows_aligns_history_future_nwp_and_target():
    panel = make_window_panel()

    windows = preprocessing.build_day_ahead_windows(panel, nwp_columns=["var78"])

    assert windows["history_power"].shape == (2, 24, 1)
    assert windows["future_nwp_raw"].shape == (2, 24, 1)
    assert windows["future_calendar"].shape == (2, 24, 4)
    assert windows["target_power"].shape == (2, 24)
    assert "origin_timestamp" not in windows
    assert "target_timestamp" not in windows
    assert pd.to_datetime(
        windows["origin_timestamp_utc"][0], unit="ns"
    ) == pd.Timestamp(
        "2020-01-02 00:00"
    )
    expected_targets = pd.date_range("2020-01-02 01:00", periods=24, freq="h").asi8
    np.testing.assert_array_equal(windows["target_timestamp_utc"][0], expected_targets)
    assert windows["history_power"][0, :, 0].tolist() == list(range(24))
    assert windows["target_power"][0].tolist() == list(range(24, 48))


def test_assign_target_day_splits_uses_inclusive_label_day_boundaries():
    origins = pd.to_datetime(
        ["2013-12-31", "2014-01-01", "2014-04-01", "2014-07-01"]
    ).asi8
    split_dates = {
        "train": ["2012-04-02", "2013-12-31"],
        "validation": ["2014-01-01", "2014-03-31"],
        "test": ["2014-04-01", "2014-06-30"],
    }

    masks = preprocessing.assign_target_day_splits(origins, split_dates)

    assert masks["train"].tolist() == [True, False, False, False]
    assert masks["validation"].tolist() == [False, True, False, False]
    assert masks["test"].tolist() == [False, False, True, False]


def test_nwp_scaler_ignores_validation_and_test_values():
    train = np.array([[[0.0], [2.0]]], dtype=np.float32)
    validation = np.array([[[1000.0], [2000.0]]], dtype=np.float32)

    scaler = preprocessing.fit_nwp_scaler(train, ["var78"])
    transformed = preprocessing.apply_nwp_scaler(validation, scaler)

    assert scaler["feature_names"] == ["var78"]
    assert scaler["mean"] == [1.0]
    assert scaler["scale"] == [1.0]
    assert transformed[0, 0, 0] == pytest.approx(999.0)


def test_save_processed_artifacts_round_trips_arrays_and_utf8_json(tmp_path):
    split_windows = {
        "train": {
            "history_power": np.arange(24, dtype=np.float32).reshape(1, 24, 1),
            "origin_timestamp_utc": np.array([pd.Timestamp("2020-01-02").value]),
        }
    }
    scaler = {"feature_names": ["var78"], "mean": [1.0], "scale": [2.0]}
    metadata = {"sample_counts": {"train": 1}, "说明": "训练集专用标准化"}

    preprocessing.save_processed_artifacts(
        tmp_path, split_windows, scaler=scaler, metadata=metadata
    )

    with np.load(tmp_path / "train.npz") as saved:
        np.testing.assert_array_equal(
            saved["history_power"], split_windows["train"]["history_power"]
        )
    assert json.loads((tmp_path / "nwp_scaler.json").read_text(encoding="utf-8")) == scaler
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == metadata
