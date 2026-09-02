import json
from pathlib import Path

import numpy as np
import yaml

import scripts.run_exploratory_analysis as exploratory_analysis


DAY_NS = 86_400_000_000_000


def _make_split(date_count: int, start_ns: int) -> dict[str, np.ndarray]:
    horizon = 4
    origins = np.repeat(
        start_ns + np.arange(date_count, dtype=np.int64) * DAY_NS,
        3,
    )
    zones = np.tile(np.array([1, 2, 3], dtype=np.int16), date_count)
    sample_count = origins.size
    history = np.zeros((sample_count, 4, 1), dtype=np.float32)
    target = np.broadcast_to(
        np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32),
        (sample_count, horizon),
    ).copy()
    nwp = np.zeros((sample_count, horizon, 12), dtype=np.float32)
    nwp[:, :, 8] = 1.0
    target_times = origins[:, None] + (
        np.arange(1, horizon + 1, dtype=np.int64)[None, :] * 3_600_000_000_000
    )
    return {
        "history_power": history,
        "future_nwp_raw": nwp,
        "future_nwp_scaled": nwp.copy(),
        "future_calendar": np.zeros((sample_count, horizon, 4), dtype=np.float32),
        "target_power": target,
        "zone_id": zones,
        "origin_timestamp_utc": origins,
        "target_timestamp_utc": target_times,
    }


def _write_fixture(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    processed = root / "data/processed/day_ahead"
    processed.mkdir(parents=True)
    metadata = {
        "dataset": "synthetic fixture",
        "time_standard": "UTC",
        "nwp_feature_names": [
            "var78", "var79", "var134", "var157", "var164", "var165",
            "var166", "var167", "var169_hourly", "var175_hourly",
            "var178_hourly", "var228_hourly",
        ],
        "calendar_feature_names": [
            "utc_hour_sin", "utc_hour_cos", "day_of_year_sin", "day_of_year_cos"
        ],
        "zones": [1, 2, 3],
    }
    (processed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    train = _make_split(6, 0)
    validation = _make_split(4, 100 * DAY_NS)
    test = _make_split(16, 200 * DAY_NS)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        np.savez_compressed(processed / f"{name}.npz", **split)

    config = {
        "data": {"processed_dir": "data/processed/day_ahead"},
        "exploratory_analysis": {
            "post_hoc_exploratory": True,
            "high_power_quantile": 0.75,
            "ramp_quantile": 0.90,
            "bootstrap_block_lengths_days": [3, 7, 14],
            "primary_block_length_days": 7,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20260901,
        },
    }
    config_dir = root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "day_ahead.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    actual = test["target_power"]
    errors = {
        "daily_persistence": 0.30,
        "seasonal_persistence_7d": 0.25,
        "xgboost": 0.10,
        "ordinary_lstm": 0.09,
        "attention_lstm": 0.11,
        "history_only": 0.20,
        "nwp_only": 0.15,
    }
    predictions = {
        name: np.stack(
            [actual + error, actual + error + (0.01 if seed_index else 0.0)]
            if name not in {"daily_persistence", "seasonal_persistence_7d"}
            else [actual + error]
        ).astype(np.float32)
        for name, error in errors.items()
        for seed_index in [0]
    }
    return config_path, predictions


def test_exploratory_analysis_writes_paired_results_and_figures(tmp_path):
    config_path, predictions = _write_fixture(tmp_path)
    artifact_dir = tmp_path / "outputs/analysis"
    figure_dir = tmp_path / "outputs/figures"

    results = exploratory_analysis.run_analysis(
        config_path,
        predictions_by_model=predictions,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        n_resamples=20,
    )

    assert results["analysis_status"] == "post_hoc_exploratory"
    assert results["main_model_reselected"] is False
    assert results["thresholds"]["derived_from"] == "training_target_power"
    assert set(results["models"]) == set(predictions)
    assert set(results["paired_bootstrap"]) == {
        "ordinary_lstm_vs_xgboost",
        "attention_lstm_vs_ordinary_lstm",
        "history_only_vs_ordinary_lstm",
        "nwp_only_vs_ordinary_lstm",
    }
    for comparison in results["paired_bootstrap"].values():
        assert set(comparison) == {"mae", "rmse"}
        assert set(comparison["rmse"]) == {"3", "7", "14"}
        assert comparison["rmse"]["7"]["n_resamples"] == 20
        assert comparison["rmse"]["7"]["difference"] == "candidate_minus_reference"

    saved = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == results
    for name in (
        "unified_model_comparison.png",
        "model_horizon_rmse.png",
        "model_zone_rmse.png",
    ):
        assert (figure_dir / name).stat().st_size > 0
