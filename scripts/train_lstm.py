"""Train and evaluate the fixed ordinary Seq2Seq LSTM experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.baselines import daily_persistence  # noqa: E402
from renewable_forecasting.lstm_model import (  # noqa: E402
    Seq2SeqLSTM,
    build_sequence_arrays,
)
from renewable_forecasting.lstm_training import (  # noqa: E402
    TrainingHistory,
    fit_lstm,
    predict_lstm,
    set_reproducible_seed,
)
from renewable_forecasting.metrics import summarize_forecast_errors  # noqa: E402
from renewable_forecasting.xgboost_model import aggregate_seed_metrics  # noqa: E402


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"processed split not found: {path}")
    with np.load(path) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _resolved_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LSTM configuration requests CUDA, but CUDA is unavailable")
    return device


def _irradiance_active_mask(
    split: dict[str, np.ndarray],
    irradiance_index: int,
) -> np.ndarray:
    return split["future_nwp_raw"][:, :, irradiance_index] > 0.0


def _evaluate_prediction(
    split: dict[str, np.ndarray],
    prediction: np.ndarray,
    irradiance_index: int,
) -> dict[str, object]:
    return dict(
        summarize_forecast_errors(
            split["target_power"],
            prediction,
            split["zone_id"],
            irradiance_active_mask=_irradiance_active_mask(
                split,
                irradiance_index,
            ),
        )
    )


def _history_record(history: TrainingHistory) -> dict[str, object]:
    return {
        "train_losses": list(history.train_losses),
        "validation_losses": list(history.validation_losses),
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "epochs_completed": history.epochs_completed,
        "stopped_early": history.stopped_early,
    }


def _portable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _comparison_metrics(
    aggregate: dict[str, object],
    project_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_path = project_root / "artifacts" / "baselines" / "metrics.json"
    xgboost_path = project_root / "artifacts" / "xgboost" / "metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline metrics not found: {baseline_path}")
    if not xgboost_path.is_file():
        raise FileNotFoundError(f"XGBoost metrics not found: {xgboost_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    xgboost = json.loads(xgboost_path.read_text(encoding="utf-8"))

    baseline_comparison: dict[str, object] = {}
    xgboost_comparison: dict[str, object] = {}
    for split_name in ("validation", "test"):
        baseline_comparison[split_name] = {}
        xgboost_comparison[split_name] = {}
        for scope_name in ("overall", "irradiance_active"):
            lstm_rmse = float(
                aggregate[split_name][scope_name]["rmse"]["mean"]
            )
            baseline_rmse = float(
                baseline[split_name]["daily_persistence"][scope_name]["rmse"]
            )
            xgboost_rmse = float(
                xgboost["aggregate"][split_name][scope_name]["rmse"]["mean"]
            )
            baseline_comparison[split_name][scope_name] = {
                "daily_persistence_rmse": baseline_rmse,
                "lstm_mean_rmse": lstm_rmse,
                "rmse_reduction_percent": (
                    100.0 * (baseline_rmse - lstm_rmse) / baseline_rmse
                ),
            }
            xgboost_comparison[split_name][scope_name] = {
                "xgboost_mean_rmse": xgboost_rmse,
                "lstm_mean_rmse": lstm_rmse,
                "lstm_rmse_reduction_percent": (
                    100.0 * (xgboost_rmse - lstm_rmse) / xgboost_rmse
                ),
            }
    return baseline_comparison, xgboost_comparison


def _plot_training_curves(
    histories: list[tuple[int, TrainingHistory]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for seed, history in histories:
        epochs = np.arange(1, history.epochs_completed + 1)
        axes[0].plot(epochs, history.train_losses, label=f"Seed {seed}")
        axes[1].plot(epochs, history.validation_losses, label=f"Seed {seed}")
        axes[1].scatter(
            history.best_epoch,
            history.best_validation_loss,
            s=30,
            zorder=3,
        )
    axes[0].set_title("Training MSE")
    axes[1].set_title("Validation MSE (best epoch marked)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_forecast_comparison(
    test_split: dict[str, np.ndarray],
    lstm_prediction: np.ndarray,
    xgboost_prediction: np.ndarray,
    output_path: Path,
) -> None:
    if xgboost_prediction.shape != lstm_prediction.shape:
        raise ValueError("XGBoost and LSTM test predictions must have identical shapes")
    sample_index = 0
    timestamps = pd.to_datetime(
        test_split["target_timestamp_utc"][sample_index],
        unit="ns",
        utc=True,
    )
    persistence = daily_persistence(test_split["history_power"])
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(
        timestamps,
        test_split["target_power"][sample_index],
        color="#111827",
        linewidth=2.2,
        label="Observed",
    )
    axis.plot(
        timestamps,
        persistence[sample_index],
        color="#dc2626",
        linewidth=1.4,
        label="Daily persistence",
    )
    axis.plot(
        timestamps,
        xgboost_prediction[sample_index],
        color="#2563eb",
        linewidth=1.7,
        label="XGBoost mean",
    )
    axis.plot(
        timestamps,
        lstm_prediction[sample_index],
        color="#059669",
        linewidth=1.9,
        label="LSTM mean",
    )
    axis.set_title(
        f"Zone {int(test_split['zone_id'][sample_index])} test forecast (UTC): "
        f"{timestamps[0]:%Y-%m-%d}"
    )
    axis.set_xlabel("Forecast valid time (UTC)")
    axis.set_ylabel("Normalized power")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_training(
    config_path: str | Path,
    artifact_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train fixed seeds, selecting checkpoints only by validation MSE."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    protocol: dict[str, Any] = config["lstm"]
    if protocol["loss"] != "mse":
        raise ValueError("ordinary LSTM training currently requires MSE loss")
    processed_dir = _resolve_path(project_root, config["data"]["processed_dir"])
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"processed metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("time_standard") != "UTC":
        raise ValueError("processed data must declare UTC as its time standard")

    nwp_feature_names = list(metadata["nwp_feature_names"])
    calendar_feature_names = list(metadata["calendar_feature_names"])
    zone_categories = [int(zone) for zone in metadata["zones"]]
    try:
        irradiance_index = nwp_feature_names.index("var169_hourly")
    except ValueError as error:
        raise ValueError("processed data must include var169_hourly") from error

    artifact_dir = _resolve_path(
        project_root,
        artifact_dir if artifact_dir is not None else Path("artifacts") / "lstm",
    )
    checkpoint_dir = _resolve_path(
        project_root,
        checkpoint_dir
        if checkpoint_dir is not None
        else Path("artifacts") / "checkpoints",
    )
    figure_dir = _resolve_path(
        project_root,
        figure_dir if figure_dir is not None else Path("reports") / "figures",
    )
    for directory in (artifact_dir, checkpoint_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_split = _load_split(processed_dir / "train.npz")
    validation_split = _load_split(processed_dir / "validation.npz")
    test_split = _load_split(processed_dir / "test.npz")
    train_arrays = build_sequence_arrays(train_split, zone_categories)
    validation_arrays = build_sequence_arrays(validation_split, zone_categories)
    test_arrays = build_sequence_arrays(test_split, zone_categories)
    future_feature_count = train_arrays.future_covariates.shape[2]
    if (
        validation_arrays.future_covariates.shape[2] != future_feature_count
        or test_arrays.future_covariates.shape[2] != future_feature_count
    ):
        raise ValueError("train, validation, and test covariate schemas differ")

    device = _resolved_device(str(protocol["device"]))
    clip_values = protocol["prediction_clip"]
    clip_bounds = (float(clip_values[0]), float(clip_values[1]))
    seeds = [int(seed) for seed in config["evaluation"]["random_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation random seeds must be unique and non-empty")

    architecture = {
        "future_feature_count": future_feature_count,
        "zone_count": len(zone_categories),
        "zone_embedding_dim": int(protocol["zone_embedding_dim"]),
        "hidden_size": int(protocol["hidden_size"]),
        "num_layers": int(protocol["num_layers"]),
    }
    runs: list[dict[str, object]] = []
    histories: list[tuple[int, TrainingHistory]] = []
    validation_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    for seed in seeds:
        set_reproducible_seed(
            seed,
            deterministic_algorithms=bool(protocol["deterministic_algorithms"]),
        )
        model = Seq2SeqLSTM(**architecture)
        history = fit_lstm(
            model,
            train_arrays,
            validation_arrays,
            batch_size=int(protocol["batch_size"]),
            learning_rate=float(protocol["learning_rate"]),
            weight_decay=float(protocol["weight_decay"]),
            max_epochs=int(protocol["max_epochs"]),
            patience=int(protocol["early_stopping_patience"]),
            min_delta=float(protocol["early_stopping_min_delta"]),
            gradient_clip_norm=float(protocol["gradient_clip_norm"]),
            device=device,
            seed=seed,
            num_workers=int(protocol["num_workers"]),
        )
        validation_prediction = predict_lstm(
            model,
            validation_arrays,
            batch_size=int(protocol["batch_size"]),
            device=device,
            clip_bounds=clip_bounds,
            num_workers=int(protocol["num_workers"]),
        )
        test_prediction = predict_lstm(
            model,
            test_arrays,
            batch_size=int(protocol["batch_size"]),
            device=device,
            clip_bounds=clip_bounds,
            num_workers=int(protocol["num_workers"]),
        )
        runs.append(
            {
                "seed": seed,
                "training": _history_record(history),
                "validation": _evaluate_prediction(
                    validation_split,
                    validation_prediction,
                    irradiance_index,
                ),
                "test": _evaluate_prediction(
                    test_split,
                    test_prediction,
                    irradiance_index,
                ),
            }
        )
        histories.append((seed, history))
        validation_predictions.append(validation_prediction)
        test_predictions.append(test_prediction)
        torch.save(
            {
                "model": "ordinary_seq2seq_lstm",
                "seed": seed,
                "architecture": architecture,
                "model_state_dict": _portable_state_dict(model),
                "training_history": asdict(history),
                "zone_categories": zone_categories,
                "nwp_feature_names": nwp_feature_names,
                "calendar_feature_names": calendar_feature_names,
                "prediction_clip": clip_bounds,
                "time_standard": "UTC",
            },
            checkpoint_dir / f"lstm_seed_{seed}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    aggregate = aggregate_seed_metrics(runs)
    mean_validation_prediction = np.mean(
        np.stack(validation_predictions),
        axis=0,
    ).astype(np.float32)
    mean_test_prediction = np.mean(
        np.stack(test_predictions),
        axis=0,
    ).astype(np.float32)
    baseline_comparison, xgboost_comparison = _comparison_metrics(
        aggregate,
        project_root,
    )

    results: dict[str, object] = {
        "dataset": metadata["dataset"],
        "time_standard": "UTC",
        "model": "ordinary_seq2seq_lstm",
        "architecture": architecture,
        "feature_design": (
            "24-hour historical normalized power encoder; 24-hour decoder "
            "inputs use train-scaled NWP forecasts, four UTC calendar features, "
            "and a learned zone embedding; no teacher forcing"
        ),
        "row_counts": {
            "train": int(train_arrays.targets.shape[0]),
            "validation": int(validation_arrays.targets.shape[0]),
            "test": int(test_arrays.targets.shape[0]),
        },
        "selection_rule": {
            "metric": "validation_mse",
            "early_stopping_patience": int(protocol["early_stopping_patience"]),
            "early_stopping_min_delta": float(
                protocol["early_stopping_min_delta"]
            ),
            "best_checkpoint_restored": True,
            "test_used_for_selection": False,
        },
        "training_protocol": {
            key: protocol[key]
            for key in (
                "device",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "max_epochs",
                "gradient_clip_norm",
                "loss",
                "deterministic_algorithms",
                "num_workers",
            )
        },
        "prediction_clip": list(clip_bounds),
        "random_seeds": seeds,
        "runs": runs,
        "aggregate": aggregate,
        "baseline_comparison": baseline_comparison,
        "xgboost_comparison": xgboost_comparison,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        artifact_dir / "mean_predictions.npz",
        validation=mean_validation_prediction,
        test=mean_test_prediction,
        seeds=np.asarray(seeds, dtype=np.int32),
    )
    _plot_training_curves(histories, figure_dir / "lstm_training_curves.png")

    xgboost_predictions_path = (
        project_root / "artifacts" / "xgboost" / "mean_predictions.npz"
    )
    if not xgboost_predictions_path.is_file():
        raise FileNotFoundError(
            f"XGBoost mean predictions not found: {xgboost_predictions_path}"
        )
    with np.load(xgboost_predictions_path) as saved:
        xgboost_test_prediction = saved["test"].copy()
    _plot_forecast_comparison(
        test_split,
        mean_test_prediction,
        xgboost_test_prediction,
        figure_dir / "lstm_vs_xgboost_forecast.png",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "day_ahead.yaml",
        help="Path to the fixed experiment YAML configuration.",
    )
    args = parser.parse_args()
    results = run_training(args.config)
    for split_name in ("validation", "test"):
        overall = results["aggregate"][split_name]["overall"]
        active = results["aggregate"][split_name]["irradiance_active"]
        print(
            f"{split_name}: "
            f"RMSE={overall['rmse']['mean']:.6f} "
            f"+/- {overall['rmse']['std']:.6f}; "
            f"irradiance-active RMSE={active['rmse']['mean']:.6f} "
            f"+/- {active['rmse']['std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
