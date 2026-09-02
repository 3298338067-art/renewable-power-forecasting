"""Train and evaluate the fixed Temporal-Attention LSTM experiment."""

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
from torch.utils.data import DataLoader, TensorDataset
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.attention_lstm_model import (  # noqa: E402
    TemporalAttentionLSTM,
)
from renewable_forecasting.baselines import daily_persistence  # noqa: E402
from renewable_forecasting.lstm_model import (  # noqa: E402
    Seq2SeqLSTM,
    SequenceModelArrays,
    build_sequence_arrays,
)
from renewable_forecasting.lstm_training import (  # noqa: E402
    TrainingHistory,
    fit_lstm,
    set_reproducible_seed,
)
from renewable_forecasting.xgboost_model import aggregate_seed_metrics  # noqa: E402
from scripts.train_lstm import (  # noqa: E402
    _evaluate_prediction,
    _history_record,
    _load_split,
    _plot_training_curves,
    _portable_state_dict,
    _resolve_path,
    _resolved_device,
)


def _as_prediction_dataset(arrays: SequenceModelArrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.history_power),
        torch.from_numpy(arrays.future_covariates),
        torch.from_numpy(arrays.zone_indices),
    )


def _predict_with_attention(
    model: TemporalAttentionLSTM,
    arrays: SequenceModelArrays,
    *,
    batch_size: int,
    device: torch.device,
    clip_bounds: tuple[float, float],
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        _as_prediction_dataset(arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.to(device)
    model.eval()
    prediction_batches: list[np.ndarray] = []
    attention_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for history, future, zones in loader:
            predictions, attention_weights = model.forward_with_attention(
                history.to(device),
                future.to(device),
                zones.to(device),
            )
            prediction_batches.append(predictions.detach().cpu().numpy())
            attention_batches.append(attention_weights.detach().cpu().numpy())
    lower, upper = clip_bounds
    return (
        np.clip(
            np.concatenate(prediction_batches, axis=0),
            lower,
            upper,
        ).astype(np.float32, copy=False),
        np.concatenate(attention_batches, axis=0).astype(np.float32, copy=False),
    )


def _load_comparison_metrics(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "daily_persistence": project_root / "artifacts/baselines/metrics.json",
        "xgboost": project_root / "artifacts/xgboost/metrics.json",
        "ordinary_lstm": project_root / "artifacts/lstm/metrics.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} metrics not found: {path}")
    baseline = json.loads(paths["daily_persistence"].read_text(encoding="utf-8"))
    xgboost = json.loads(paths["xgboost"].read_text(encoding="utf-8"))
    ordinary_lstm = json.loads(
        paths["ordinary_lstm"].read_text(encoding="utf-8")
    )
    return {
        "daily_persistence": {
            split: baseline[split]["daily_persistence"]
            for split in ("validation", "test")
        },
        "xgboost": xgboost["aggregate"],
        "ordinary_lstm": ordinary_lstm["aggregate"],
    }


def _build_comparisons(
    aggregate: dict[str, Any],
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        comparisons[split_name] = {}
        for scope_name in ("overall", "irradiance_active"):
            attention_metrics = aggregate[split_name][scope_name]
            attention_rmse = float(attention_metrics["rmse"]["mean"])
            attention_mae = float(attention_metrics["mae"]["mean"])
            comparisons[split_name][scope_name] = {}
            for reference_name in (
                "daily_persistence",
                "xgboost",
                "ordinary_lstm",
            ):
                reference_metrics = reference[reference_name][split_name][scope_name]
                if reference_name == "daily_persistence":
                    reference_rmse = float(reference_metrics["rmse"])
                    reference_mae = None
                else:
                    reference_rmse = float(reference_metrics["rmse"]["mean"])
                    reference_mae = float(reference_metrics["mae"]["mean"])
                record: dict[str, float] = {
                    "reference_rmse": reference_rmse,
                    "attention_lstm_rmse": attention_rmse,
                    "rmse_reduction_percent": (
                        100.0
                        * (reference_rmse - attention_rmse)
                        / reference_rmse
                    ),
                }
                if reference_mae is not None:
                    record.update(
                        {
                            "reference_mae": reference_mae,
                            "attention_lstm_mae": attention_mae,
                            "mae_reduction_percent": (
                                100.0
                                * (reference_mae - attention_mae)
                                / reference_mae
                            ),
                        }
                    )
                comparisons[split_name][scope_name][reference_name] = record
    return comparisons


def _validation_decision(
    aggregate: dict[str, Any],
    ordinary_lstm: dict[str, Any],
) -> dict[str, Any]:
    attention = aggregate["validation"]["overall"]
    ordinary = ordinary_lstm["validation"]["overall"]
    attention_rmse = float(attention["rmse"]["mean"])
    ordinary_rmse = float(ordinary["rmse"]["mean"])
    attention_mae = float(attention["mae"]["mean"])
    ordinary_mae = float(ordinary["mae"]["mean"])
    rmse_improved = attention_rmse < ordinary_rmse
    mae_improved = attention_mae < ordinary_mae
    if rmse_improved and mae_improved:
        outcome = "validation_rmse_and_mae_improved"
    elif rmse_improved:
        outcome = "validation_rmse_improved_mae_not_improved"
    else:
        outcome = "validation_rmse_not_improved"
    return {
        "primary_metric": "validation_rmse",
        "attention_improved": rmse_improved,
        "validation_rmse_improved": rmse_improved,
        "validation_mae_improved": mae_improved,
        "ordinary_lstm_rmse": ordinary_rmse,
        "attention_lstm_rmse": attention_rmse,
        "ordinary_lstm_mae": ordinary_mae,
        "attention_lstm_mae": attention_mae,
        "outcome": outcome,
        "test_metrics_consulted": False,
    }


def _plot_forecast_comparison(
    test_split: dict[str, np.ndarray],
    ordinary_prediction: np.ndarray,
    attention_prediction: np.ndarray,
    output_path: Path,
) -> None:
    if ordinary_prediction.shape != attention_prediction.shape:
        raise ValueError("ordinary and attention predictions must have equal shapes")
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
        linewidth=1.3,
        label="Daily persistence",
    )
    axis.plot(
        timestamps,
        ordinary_prediction[sample_index],
        color="#2563eb",
        linewidth=1.8,
        label="Ordinary LSTM mean",
    )
    axis.plot(
        timestamps,
        attention_prediction[sample_index],
        color="#7c3aed",
        linewidth=1.9,
        label="Attention-LSTM mean",
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


def _plot_attention_heatmap(
    test_split: dict[str, np.ndarray],
    attention_weights: np.ndarray,
    output_path: Path,
) -> None:
    sample_index = 0
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(
        attention_weights[sample_index],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
    )
    ticks = np.array([0, 5, 11, 17, 23])
    axis.set_xticks(ticks, ticks + 1)
    axis.set_yticks(ticks, ticks + 1)
    axis.set_xlabel("Historical encoder hour")
    axis.set_ylabel("Future forecast hour")
    axis.set_title(
        "Three-seed mean temporal attention weights\n"
        "(descriptive, not causal)\n"
        f"Zone {int(test_split['zone_id'][sample_index])}, first test sample"
    )
    figure.colorbar(image, ax=axis, label="Attention weight")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_training(
    config_path: str | Path,
    artifact_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train fixed seeds and judge the model using validation RMSE."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol: dict[str, Any] = config["attention_lstm"]
    if protocol["loss"] != "mse":
        raise ValueError("Attention-LSTM training currently requires MSE loss")

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
        artifact_dir
        if artifact_dir is not None
        else Path("artifacts") / "attention_lstm",
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

    attention_architecture = {
        "future_feature_count": future_feature_count,
        "zone_count": len(zone_categories),
        "zone_embedding_dim": int(protocol["zone_embedding_dim"]),
        "hidden_size": int(protocol["hidden_size"]),
        "num_layers": int(protocol["num_layers"]),
        "attention_size": int(protocol["attention_size"]),
    }
    ordinary_protocol = config["lstm"]
    ordinary_architecture = {
        "future_feature_count": future_feature_count,
        "zone_count": len(zone_categories),
        "zone_embedding_dim": int(ordinary_protocol["zone_embedding_dim"]),
        "hidden_size": int(ordinary_protocol["hidden_size"]),
        "num_layers": int(ordinary_protocol["num_layers"]),
    }
    ordinary_model = Seq2SeqLSTM(**ordinary_architecture)
    attention_count_model = TemporalAttentionLSTM(**attention_architecture)
    parameter_counts = {
        "ordinary_lstm": sum(
            parameter.numel() for parameter in ordinary_model.parameters()
        ),
        "attention_lstm": sum(
            parameter.numel() for parameter in attention_count_model.parameters()
        ),
    }
    del ordinary_model, attention_count_model

    device = _resolved_device(str(protocol["device"]))
    clip_values = protocol["prediction_clip"]
    clip_bounds = (float(clip_values[0]), float(clip_values[1]))
    seeds = [int(seed) for seed in config["evaluation"]["random_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation random seeds must be unique and non-empty")

    runs: list[dict[str, object]] = []
    histories: list[tuple[int, TrainingHistory]] = []
    validation_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    validation_attention: list[np.ndarray] = []
    test_attention: list[np.ndarray] = []
    for seed in seeds:
        set_reproducible_seed(
            seed,
            deterministic_algorithms=bool(protocol["deterministic_algorithms"]),
        )
        model = TemporalAttentionLSTM(**attention_architecture)
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
        validation_prediction, validation_weights = _predict_with_attention(
            model,
            validation_arrays,
            batch_size=int(protocol["batch_size"]),
            device=device,
            clip_bounds=clip_bounds,
            num_workers=int(protocol["num_workers"]),
        )
        test_prediction, test_weights = _predict_with_attention(
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
        validation_attention.append(validation_weights)
        test_attention.append(test_weights)
        torch.save(
            {
                "model": "temporal_attention_lstm",
                "seed": seed,
                "architecture": attention_architecture,
                "model_state_dict": _portable_state_dict(model),
                "training_history": asdict(history),
                "zone_categories": zone_categories,
                "nwp_feature_names": nwp_feature_names,
                "calendar_feature_names": calendar_feature_names,
                "prediction_clip": clip_bounds,
                "time_standard": "UTC",
            },
            checkpoint_dir / f"attention_lstm_seed_{seed}.pt",
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
    mean_validation_attention = np.mean(
        np.stack(validation_attention),
        axis=0,
    ).astype(np.float32)
    mean_test_attention = np.mean(
        np.stack(test_attention),
        axis=0,
    ).astype(np.float32)
    reference = _load_comparison_metrics(project_root)
    comparisons = _build_comparisons(aggregate, reference)
    validation_decision = _validation_decision(
        aggregate,
        reference["ordinary_lstm"],
    )

    results: dict[str, object] = {
        "dataset": metadata["dataset"],
        "time_standard": "UTC",
        "model": "temporal_attention_lstm",
        "architecture": attention_architecture,
        "parameter_counts": parameter_counts,
        "feature_design": (
            "Same past-power, train-scaled future NWP, UTC calendar, and zone "
            "inputs as ordinary LSTM; additive attention over all 24 historical "
            "encoder states before each future decoder step; no teacher forcing"
        ),
        "row_counts": {
            "train": int(train_arrays.targets.shape[0]),
            "validation": int(validation_arrays.targets.shape[0]),
            "test": int(test_arrays.targets.shape[0]),
        },
        "selection_rule": {
            "metric": "validation_rmse",
            "fixed_architecture_no_candidate_search": True,
            "test_used_for_selection": False,
            "best_checkpoint_metric": "validation_mse",
            "early_stopping_patience": int(protocol["early_stopping_patience"]),
            "early_stopping_min_delta": float(
                protocol["early_stopping_min_delta"]
            ),
        },
        "validation_decision": validation_decision,
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
        "comparisons": comparisons,
        "attention_interpretation_note": (
            "Attention weights are descriptive model quantities, not causal "
            "feature importance."
        ),
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
    np.savez_compressed(
        artifact_dir / "mean_attention_weights.npz",
        validation=mean_validation_attention,
        test=mean_test_attention,
        seeds=np.asarray(seeds, dtype=np.int32),
    )
    _plot_training_curves(
        histories,
        figure_dir / "attention_lstm_training_curves.png",
    )

    ordinary_predictions_path = (
        project_root / "artifacts/lstm/mean_predictions.npz"
    )
    if not ordinary_predictions_path.is_file():
        raise FileNotFoundError(
            f"ordinary LSTM predictions not found: {ordinary_predictions_path}"
        )
    with np.load(ordinary_predictions_path) as saved:
        ordinary_test_prediction = saved["test"].copy()
    _plot_forecast_comparison(
        test_split,
        ordinary_test_prediction,
        mean_test_prediction,
        figure_dir / "attention_lstm_vs_lstm_forecast.png",
    )
    _plot_attention_heatmap(
        test_split,
        mean_test_attention,
        figure_dir / "attention_lstm_weights_heatmap.png",
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
    decision = results["validation_decision"]
    print(
        "Validation decision: "
        f"{decision['outcome']} "
        f"(Attention RMSE={decision['attention_lstm_rmse']:.6f}, "
        f"ordinary LSTM RMSE={decision['ordinary_lstm_rmse']:.6f})"
    )
    for split_name in ("validation", "test"):
        overall = results["aggregate"][split_name]["overall"]
        print(
            f"{split_name}: MAE={overall['mae']['mean']:.6f} "
            f"+/- {overall['mae']['std']:.6f}; "
            f"RMSE={overall['rmse']['mean']:.6f} "
            f"+/- {overall['rmse']['std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
