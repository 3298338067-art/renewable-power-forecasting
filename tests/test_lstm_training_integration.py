import json
from pathlib import Path

import numpy as np
import yaml

import scripts.train_lstm as train_lstm


def _make_split(sample_count: int, start_day: str) -> dict[str, np.ndarray]:
    horizon = 24
    zones = np.resize(np.array([1, 2, 3], dtype=np.int16), sample_count)
    history = np.linspace(
        0.0,
        0.8,
        sample_count * horizon,
        dtype=np.float32,
    ).reshape(sample_count, horizon, 1)
    future_nwp = np.zeros((sample_count, horizon, 12), dtype=np.float32)
    future_nwp[:, :12, 8] = 1.0
    future_nwp[:, :, 0] = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    future_calendar = np.zeros((sample_count, horizon, 4), dtype=np.float32)
    future_calendar[:, :, 0] = np.sin(
        2.0 * np.pi * np.arange(horizon, dtype=np.float32) / horizon
    )
    target = np.clip(0.6 * future_nwp[:, :, 0] + 0.1, 0.0, 1.0)
    first = np.datetime64(start_day, "ns") + np.timedelta64(1, "h")
    timestamps = first + np.arange(horizon) * np.timedelta64(1, "h")
    return {
        "history_power": history,
        "future_nwp_raw": future_nwp,
        "future_nwp_scaled": future_nwp.copy(),
        "future_calendar": future_calendar,
        "target_power": np.broadcast_to(target, (sample_count, horizon)).copy(),
        "zone_id": zones,
        "target_timestamp_utc": np.broadcast_to(
            timestamps.astype("datetime64[ns]").astype(np.int64),
            (sample_count, horizon),
        ).copy(),
    }


def _write_fixture_project(root: Path) -> Path:
    processed_dir = root / "data" / "processed" / "day_ahead"
    processed_dir.mkdir(parents=True)
    metadata = {
        "dataset": "synthetic solar fixture",
        "time_standard": "UTC",
        "nwp_feature_names": [
            "var78",
            "var79",
            "var134",
            "var157",
            "var164",
            "var165",
            "var166",
            "var167",
            "var169_hourly",
            "var175_hourly",
            "var178_hourly",
            "var228_hourly",
        ],
        "calendar_feature_names": [
            "utc_hour_sin",
            "utc_hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ],
        "zones": [1, 2, 3],
    }
    (processed_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    for name, count, start in (
        ("train", 6, "2012-04-02"),
        ("validation", 3, "2014-01-01"),
        ("test", 3, "2014-04-01"),
    ):
        np.savez_compressed(
            processed_dir / f"{name}.npz",
            **_make_split(count, start),
        )

    baseline_dir = root / "artifacts" / "baselines"
    xgboost_dir = root / "artifacts" / "xgboost"
    baseline_dir.mkdir(parents=True)
    xgboost_dir.mkdir(parents=True)
    comparison_metrics = {
        split: {
            "overall": {"rmse": overall},
            "irradiance_active": {"rmse": active},
        }
        for split, overall, active in (
            ("validation", 0.5, 0.4),
            ("test", 0.45, 0.35),
        )
    }
    (baseline_dir / "metrics.json").write_text(
        json.dumps(
            {
                split: {"daily_persistence": values}
                for split, values in comparison_metrics.items()
            }
        ),
        encoding="utf-8",
    )
    (xgboost_dir / "metrics.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    split: {
                        scope: {"rmse": {"mean": 0.3}}
                        for scope in ("overall", "irradiance_active")
                    }
                    for split in ("validation", "test")
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        xgboost_dir / "mean_predictions.npz",
        validation=np.full((3, 24), 0.3, dtype=np.float32),
        test=np.full((3, 24), 0.3, dtype=np.float32),
    )

    config = {
        "data": {"processed_dir": "data/processed/day_ahead"},
        "task": {"lookback_hours": 24, "forecast_horizon_hours": 24},
        "lstm": {
            "device": "cpu",
            "hidden_size": 4,
            "num_layers": 1,
            "zone_embedding_dim": 2,
            "batch_size": 3,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "loss": "mse",
            "prediction_clip": [0.0, 1.0],
            "deterministic_algorithms": True,
            "num_workers": 0,
        },
        "evaluation": {"random_seeds": [42, 2026]},
    }
    config_dir = root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "day_ahead.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_lstm_training_saves_reproducible_outputs(tmp_path):
    config_path = _write_fixture_project(tmp_path)
    artifact_dir = tmp_path / "outputs" / "lstm"
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    figure_dir = tmp_path / "outputs" / "figures"

    results = train_lstm.run_training(
        config_path,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
    )

    assert results["model"] == "ordinary_seq2seq_lstm"
    assert results["time_standard"] == "UTC"
    assert results["random_seeds"] == [42, 2026]
    assert results["selection_rule"]["test_used_for_selection"] is False
    assert len(results["runs"]) == 2
    assert set(results["aggregate"]) == {"validation", "test"}
    assert results["row_counts"] == {"train": 6, "validation": 3, "test": 3}
    assert set(results["baseline_comparison"]) == {"validation", "test"}
    assert set(results["xgboost_comparison"]) == {"validation", "test"}

    saved_results = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_results == results
    with np.load(artifact_dir / "mean_predictions.npz") as saved:
        assert saved["validation"].shape == (3, 24)
        assert saved["test"].shape == (3, 24)
        np.testing.assert_array_equal(saved["seeds"], [42, 2026])

    for seed in (42, 2026):
        checkpoint = checkpoint_dir / f"lstm_seed_{seed}.pt"
        assert checkpoint.is_file()
        assert checkpoint.stat().st_size > 0
    for figure_name in (
        "lstm_training_curves.png",
        "lstm_vs_xgboost_forecast.png",
    ):
        figure = figure_dir / figure_name
        assert figure.is_file()
        assert figure.stat().st_size > 0
