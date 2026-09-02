from pathlib import Path

import numpy as np
import pytest

import scripts.preprocess_data as preprocess_data


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "day_ahead.yaml"
RAW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "gefcom2014"
    / "solar"
    / "Solar"
    / "Task 15"
    / "predictors15.csv"
)


@pytest.mark.skipif(not RAW_PATH.is_file(), reason="official local dataset is unavailable")
def test_official_preprocessing_outputs_expected_split_counts(tmp_path):
    output_dir = tmp_path / "processed"
    metadata = preprocess_data.run_pipeline(
        CONFIG_PATH,
        output_dir=output_dir,
        interim_path=tmp_path / "panel.csv.gz",
    )

    assert metadata["sample_counts"] == {
        "train": 1917,
        "validation": 270,
        "test": 273,
    }
    assert metadata["zones"] == [1, 2, 3]
    assert metadata["lookback_hours"] == 24
    assert metadata["forecast_horizon_hours"] == 24
    assert metadata["time_standard"] == "UTC"
    assert metadata["timestamp_fields"] == {
        "origin_timestamp_utc": "forecast issue time in UTC",
        "target_timestamp_utc": "forecast valid time in UTC",
    }

    for split_name in ("train", "validation", "test"):
        with np.load(output_dir / f"{split_name}.npz") as split:
            assert np.isfinite(split["future_nwp_scaled"]).all()
            assert split["history_power"].shape[1:] == (24, 1)
            assert split["future_nwp_raw"].shape[1:] == (24, 12)
            assert split["future_nwp_scaled"].shape[1:] == (24, 12)
            assert split["future_calendar"].shape[1:] == (24, 4)
            assert split["target_power"].shape[1:] == (24,)
            assert "origin_timestamp" not in split.files
            assert "target_timestamp" not in split.files
            expected_target_timestamps = split["origin_timestamp_utc"][:, None] + (
                np.arange(1, 25, dtype=np.int64)[None, :] * 3_600_000_000_000
            )
            np.testing.assert_array_equal(
                split["target_timestamp_utc"], expected_target_timestamps
            )
            if split_name == "train":
                np.testing.assert_allclose(
                    split["future_nwp_scaled"].mean(
                        axis=(0, 1), dtype=np.float64
                    ),
                    0.0,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    split["future_nwp_scaled"].std(
                        axis=(0, 1), dtype=np.float64
                    ),
                    1.0,
                    atol=2e-6,
                )
