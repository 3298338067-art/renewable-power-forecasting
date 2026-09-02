import json
from pathlib import Path

import numpy as np
import torch
import yaml

from renewable_forecasting.attention_lstm_model import TemporalAttentionLSTM
import scripts.train_attention_lstm as train_attention_lstm


NWP_NAMES = [
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
]


def _make_split(sample_count: int, start_day: str) -> dict[str, np.ndarray]:
    horizon = 24
    zones = np.resize(np.array([1, 2, 3], dtype=np.int16), sample_count)
    history = np.linspace(
        0.0,
        0.8,
        sample_count * horizon,
        dtype=np.float32,
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
    return {
        "history_power": history,
        "future_nwp_raw": nwp,
        "future_nwp_scaled": nwp.copy(),
        "future_calendar": calendar,
        "target_power": target,
        "zone_id": zones,
        "target_timestamp_utc": np.broadcast_to(
            timestamps.astype("datetime64[ns]").astype(np.int64),
            (sample_count, horizon),
        ).copy(),
    }


def _write_comparison_metrics(root: Path) -> None:
    for name in ("baselines", "xgboost", "lstm"):
        (root / "artifacts" / name).mkdir(parents=True)
    baseline = {
        split: {
            "daily_persistence": {
                "overall": {"rmse": 0.5},
                "irradiance_active": {"rmse": 0.4},
            }
        }
        for split in ("validation", "test")
    }
    (root / "artifacts/baselines/metrics.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    for name, rmse in (("xgboost", 0.3), ("lstm", 0.25)):
        metrics = {
            "aggregate": {
                split: {
                    scope: {
                        "mae": {"mean": rmse - 0.05, "std": 0.01},
                        "rmse": {"mean": rmse, "std": 0.01},
                        "nrmse_capacity_percent": {
                            "mean": 100.0 * rmse,
                            "std": 1.0,
                        },
                    }
                    for scope in ("overall", "irradiance_active")
                }
                for split in ("validation", "test")
            }
        }
        (root / f"artifacts/{name}/metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    np.savez_compressed(
        root / "artifacts/lstm/mean_predictions.npz",
        validation=np.full((3, 24), 0.25, dtype=np.float32),
        test=np.full((3, 24), 0.25, dtype=np.float32),
    )


def _write_fixture_project(root: Path) -> Path:
    processed = root / "data" / "processed" / "day_ahead"
    processed.mkdir(parents=True)
    metadata = {
        "dataset": "synthetic solar fixture",
        "time_standard": "UTC",
        "nwp_feature_names": NWP_NAMES,
        "calendar_feature_names": [
            "utc_hour_sin",
            "utc_hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ],
        "zones": [1, 2, 3],
    }
    (processed / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    for name, count, start in (
        ("train", 6, "2012-04-02"),
        ("validation", 3, "2014-01-01"),
        ("test", 3, "2014-04-01"),
    ):
        np.savez_compressed(processed / f"{name}.npz", **_make_split(count, start))
    _write_comparison_metrics(root)

    config = {
        "data": {"processed_dir": "data/processed/day_ahead"},
        "task": {"lookback_hours": 24, "forecast_horizon_hours": 24},
        "lstm": {
            "hidden_size": 4,
            "num_layers": 1,
            "zone_embedding_dim": 2,
        },
        "attention_lstm": {
            "device": "cpu",
            "hidden_size": 4,
            "num_layers": 1,
            "zone_embedding_dim": 2,
            "attention_size": 4,
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


def test_attention_training_saves_fair_comparison_outputs(tmp_path):
    config_path = _write_fixture_project(tmp_path)
    artifact_dir = tmp_path / "outputs" / "attention_lstm"
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    figure_dir = tmp_path / "outputs" / "figures"

    results = train_attention_lstm.run_training(
        config_path,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
    )

    assert results["model"] == "temporal_attention_lstm"
    assert results["time_standard"] == "UTC"
    assert results["random_seeds"] == [42, 2026]
    assert results["selection_rule"]["metric"] == "validation_rmse"
    assert results["selection_rule"]["test_used_for_selection"] is False
    assert results["row_counts"] == {"train": 6, "validation": 3, "test": 3}
    assert results["parameter_counts"]["attention_lstm"] > (
        results["parameter_counts"]["ordinary_lstm"]
    )
    expected_improvement = (
        results["aggregate"]["validation"]["overall"]["rmse"]["mean"]
        < 0.25
    )
    assert results["validation_decision"]["attention_improved"] is expected_improvement

    saved_results = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_results == results
    with np.load(artifact_dir / "mean_predictions.npz") as saved:
        assert saved["validation"].shape == (3, 24)
        assert saved["test"].shape == (3, 24)
        np.testing.assert_array_equal(saved["seeds"], [42, 2026])
    with np.load(artifact_dir / "mean_attention_weights.npz") as saved:
        assert saved["validation"].shape == (3, 24, 24)
        assert saved["test"].shape == (3, 24, 24)
        np.testing.assert_allclose(saved["test"].sum(axis=2), 1.0, atol=1e-6)

    for seed in (42, 2026):
        checkpoint = checkpoint_dir / f"attention_lstm_seed_{seed}.pt"
        assert checkpoint.is_file()
        assert checkpoint.stat().st_size > 0
        saved_checkpoint = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        restored_model = TemporalAttentionLSTM(
            **saved_checkpoint["architecture"]
        )
        restored_model.load_state_dict(saved_checkpoint["model_state_dict"])
    for figure_name in (
        "attention_lstm_training_curves.png",
        "attention_lstm_vs_lstm_forecast.png",
        "attention_lstm_weights_heatmap.png",
    ):
        figure = figure_dir / figure_name
        assert figure.is_file()
        assert figure.stat().st_size > 0
