import json
from pathlib import Path

import numpy as np
import torch
import yaml

from renewable_forecasting.lstm_model import CovariateOnlyLSTM, Seq2SeqLSTM
import scripts.train_lstm_ablations as train_lstm_ablations


def _make_split(sample_count: int, start_day: str) -> dict[str, np.ndarray]:
    horizon = 24
    zones = np.resize(np.array([1, 2, 3], dtype=np.int16), sample_count)
    history = np.linspace(
        0.0, 0.8, sample_count * horizon, dtype=np.float32
    ).reshape(sample_count, horizon, 1)
    nwp = np.zeros((sample_count, horizon, 12), dtype=np.float32)
    nwp[:, :12, 8] = 1.0
    nwp[:, :, 0] = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    calendar = np.zeros((sample_count, horizon, 4), dtype=np.float32)
    calendar[:, :, 0] = np.sin(
        2.0 * np.pi * np.arange(horizon, dtype=np.float32) / horizon
    )
    target = np.clip(0.6 * nwp[:, :, 0] + 0.1, 0.0, 1.0)
    first = np.datetime64(start_day, "ns") + np.timedelta64(1, "h")
    timestamps = first + np.arange(horizon) * np.timedelta64(1, "h")
    origins = np.datetime64(start_day, "ns") + np.arange(sample_count) * np.timedelta64(1, "D")
    return {
        "history_power": history,
        "future_nwp_raw": nwp,
        "future_nwp_scaled": nwp.copy(),
        "future_calendar": calendar,
        "target_power": np.broadcast_to(target, (sample_count, horizon)).copy(),
        "zone_id": zones,
        "origin_timestamp_utc": origins.astype(np.int64),
        "target_timestamp_utc": np.broadcast_to(
            timestamps.astype("datetime64[ns]").astype(np.int64),
            (sample_count, horizon),
        ).copy(),
    }


def _write_fixture_project(root: Path) -> Path:
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
    for name, count, start in (
        ("train", 6, "2012-04-02"),
        ("validation", 3, "2014-01-01"),
        ("test", 3, "2014-04-01"),
    ):
        np.savez_compressed(processed / f"{name}.npz", **_make_split(count, start))

    ordinary_dir = root / "artifacts/lstm"
    ordinary_dir.mkdir(parents=True)
    ordinary_metrics = {
        "model": "ordinary_seq2seq_lstm",
        "aggregate": {
            split: {
                scope: {
                    "mae": {"mean": 0.2, "std": 0.01},
                    "rmse": {"mean": 0.3, "std": 0.01},
                }
                for scope in ("overall", "irradiance_active")
            }
            for split in ("validation", "test")
        },
    }
    (ordinary_dir / "metrics.json").write_text(
        json.dumps(ordinary_metrics), encoding="utf-8"
    )

    config = {
        "data": {"processed_dir": "data/processed/day_ahead"},
        "task": {"lookback_hours": 24, "forecast_horizon_hours": 24},
        "lstm": {
            "device": "cpu", "hidden_size": 4, "num_layers": 1,
            "zone_embedding_dim": 2, "batch_size": 3,
            "learning_rate": 0.01, "weight_decay": 0.0, "max_epochs": 2,
            "early_stopping_patience": 2, "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 1.0, "loss": "mse",
            "prediction_clip": [0.0, 1.0], "deterministic_algorithms": True,
            "num_workers": 0,
        },
        "lstm_ablations": {
            "post_hoc_exploratory": True,
            "variants": ["history_only", "nwp_only"],
            "full_input_control_metrics": "artifacts/lstm/metrics.json",
        },
        "evaluation": {"random_seeds": [42, 2026]},
    }
    config_dir = root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "day_ahead.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_lstm_ablation_training_saves_reusable_seed_predictions(tmp_path):
    config_path = _write_fixture_project(tmp_path)
    artifact_dir = tmp_path / "outputs/ablations"
    checkpoint_dir = tmp_path / "outputs/checkpoints"
    figure_dir = tmp_path / "outputs/figures"

    results = train_lstm_ablations.run_training(
        config_path,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
    )

    assert results["analysis_status"] == "post_hoc_exploratory"
    assert results["full_input_control"]["reused"] is True
    assert set(results["variants"]) == {"history_only", "nwp_only"}
    assert all(
        variant["selection_rule"]["test_used_for_selection"] is False
        for variant in results["variants"].values()
    )
    assert results["variants"]["history_only"]["architecture"]["future_feature_count"] == 4
    assert results["variants"]["nwp_only"]["architecture"]["future_feature_count"] == 16
    parameter_counts = {
        variant: result["parameter_count"]
        for variant, result in results["variants"].items()
    }
    assert all(count > 0 for count in parameter_counts.values())
    assert len(set(parameter_counts.values())) == 2

    with np.load(artifact_dir / "predictions.npz") as saved:
        np.testing.assert_array_equal(saved["seeds"], [42, 2026])
        for variant in ("history_only", "nwp_only"):
            assert saved[f"{variant}_validation"].shape == (2, 3, 24)
            assert saved[f"{variant}_test"].shape == (2, 3, 24)

    for variant, model_class in (
        ("history_only", Seq2SeqLSTM),
        ("nwp_only", CovariateOnlyLSTM),
    ):
        checkpoint = torch.load(
            checkpoint_dir / f"lstm_{variant}_seed_42.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = model_class(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert checkpoint["input_variant"] == variant
        assert checkpoint["time_standard"] == "UTC"
        assert (figure_dir / f"lstm_{variant}_training_curves.png").stat().st_size > 0

    saved_results = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_results == results
