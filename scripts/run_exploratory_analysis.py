"""Run fixed post-hoc diagnostics and paired date-block bootstrap analysis."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.attention_lstm_model import (  # noqa: E402
    TemporalAttentionLSTM,
)
from renewable_forecasting.baselines import (  # noqa: E402
    daily_persistence,
    seasonal_persistence,
)
from renewable_forecasting.evaluation_analysis import (  # noqa: E402
    aggregate_prediction_diagnostics,
    paired_moving_block_bootstrap,
    training_derived_masks,
)
from renewable_forecasting.lstm_model import (  # noqa: E402
    Seq2SeqLSTM,
    build_sequence_arrays,
)
from renewable_forecasting.lstm_training import predict_lstm  # noqa: E402
from renewable_forecasting.xgboost_model import build_long_horizon_table  # noqa: E402
from scripts.train_lstm import (  # noqa: E402
    _load_split,
    _resolve_path,
    _resolved_device,
)
from scripts.train_xgboost import _predict_table  # noqa: E402


MODEL_LABELS = {
    "daily_persistence": "Daily persistence",
    "seasonal_persistence_7d": "7-day persistence",
    "xgboost": "XGBoost",
    "ordinary_lstm": "Ordinary LSTM",
    "attention_lstm": "Attention-LSTM",
    "history_only": "History-only",
    "nwp_only": "NWP-only",
}
REQUIRED_MODELS = tuple(MODEL_LABELS)
COMPARISONS = {
    "ordinary_lstm_vs_xgboost": ("ordinary_lstm", "xgboost"),
    "attention_lstm_vs_ordinary_lstm": (
        "attention_lstm",
        "ordinary_lstm",
    ),
    "history_only_vs_ordinary_lstm": (
        "history_only",
        "ordinary_lstm",
    ),
    "nwp_only_vs_ordinary_lstm": ("nwp_only", "ordinary_lstm"),
}


def _load_torch_predictions(
    project_root: Path,
    test_split: dict[str, np.ndarray],
    zone_categories: list[int],
    seeds: list[int],
    *,
    checkpoint_stem: str,
    model_class: type[torch.nn.Module],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    input_variant: str = "full",
) -> np.ndarray:
    arrays = build_sequence_arrays(
        test_split,
        zone_categories,
        input_variant=input_variant,
    )
    predictions: list[np.ndarray] = []
    for seed in seeds:
        checkpoint_path = (
            project_root / "artifacts/checkpoints" / f"{checkpoint_stem}_{seed}.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(checkpoint["seed"]) != seed:
            raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
        model = model_class(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictions.append(
            predict_lstm(
                model,
                arrays,
                batch_size=batch_size,
                device=device,
                clip_bounds=tuple(
                    float(value) for value in checkpoint["prediction_clip"]
                ),
                num_workers=num_workers,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return np.stack(predictions).astype(np.float32)


def _load_xgboost_predictions(
    project_root: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    test_split: dict[str, np.ndarray],
    seeds: list[int],
) -> np.ndarray:
    table = build_long_horizon_table(
        test_split,
        metadata["nwp_feature_names"],
        metadata["calendar_feature_names"],
        metadata["zones"],
    )
    clip_bounds = tuple(float(value) for value in config["xgboost"]["prediction_clip"])
    predictions: list[np.ndarray] = []
    for seed in seeds:
        model_path = project_root / f"artifacts/xgboost/model_seed_{seed}.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        # The persisted booster is evaluated against a CPU NumPy feature table.
        # Matching the booster to that table avoids XGBoost's device fallback.
        model.set_params(device="cpu")
        predictions.append(_predict_table(model, table, clip_bounds))
        del model
        gc.collect()
    return np.stack(predictions).astype(np.float32)


def _load_ablation_predictions(
    project_root: Path,
    seeds: list[int],
    sample_count: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    path = project_root / "artifacts/lstm_ablations/predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"LSTM ablation predictions not found: {path}")
    with np.load(path) as saved:
        np.testing.assert_array_equal(
            saved["seeds"],
            np.asarray(seeds, dtype=np.int32),
        )
        result = {
            variant: saved[f"{variant}_test"].copy()
            for variant in ("history_only", "nwp_only")
        }
    expected_shape = (len(seeds), sample_count, horizon)
    for variant, values in result.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{variant} predictions have shape {values.shape}, "
                f"expected {expected_shape}"
            )
    return result


def _load_saved_predictions(
    project_root: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    test_split: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    seeds = [int(seed) for seed in config["evaluation"]["random_seeds"]]
    zone_categories = [int(zone) for zone in metadata["zones"]]
    lstm_protocol = config["lstm"]
    attention_protocol = config["attention_lstm"]
    lstm_device = _resolved_device(str(lstm_protocol["device"]))
    attention_device = _resolved_device(str(attention_protocol["device"]))
    predictions = {
        "ordinary_lstm": _load_torch_predictions(
            project_root,
            test_split,
            zone_categories,
            seeds,
            checkpoint_stem="lstm_seed",
            model_class=Seq2SeqLSTM,
            device=lstm_device,
            batch_size=int(lstm_protocol["batch_size"]),
            num_workers=int(lstm_protocol["num_workers"]),
        ),
        "attention_lstm": _load_torch_predictions(
            project_root,
            test_split,
            zone_categories,
            seeds,
            checkpoint_stem="attention_lstm_seed",
            model_class=TemporalAttentionLSTM,
            device=attention_device,
            batch_size=int(attention_protocol["batch_size"]),
            num_workers=int(attention_protocol["num_workers"]),
        ),
        "xgboost": _load_xgboost_predictions(
            project_root,
            config,
            metadata,
            test_split,
            seeds,
        ),
    }
    predictions.update(
        _load_ablation_predictions(
            project_root,
            seeds,
            test_split["target_power"].shape[0],
            test_split["target_power"].shape[1],
        )
    )
    predictions["daily_persistence"] = daily_persistence(
        test_split["history_power"]
    )[None, ...]
    interim_path = _resolve_path(project_root, config["data"]["interim_csv"])
    if not interim_path.is_file():
        raise FileNotFoundError(f"interim hourly panel not found: {interim_path}")
    panel = pd.read_csv(
        interim_path,
        usecols=["zone_id", "timestamp", "power"],
        parse_dates=["timestamp"],
    )
    predictions["seasonal_persistence_7d"] = seasonal_persistence(
        panel,
        test_split["zone_id"],
        test_split["target_timestamp_utc"],
    )[None, ...]
    return predictions


def _plot_unified_comparison(
    model_results: dict[str, object],
    output_path: Path,
) -> None:
    model_names = list(REQUIRED_MODELS)
    labels = [MODEL_LABELS[name] for name in model_names]
    mae = [
        model_results[name]["aggregate"]["overall"]["mae"]["mean"]
        for name in model_names
    ]
    rmse = [
        model_results[name]["aggregate"]["overall"]["rmse"]["mean"]
        for name in model_names
    ]
    x_positions = np.arange(len(model_names))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for axis, values, metric, color in (
        (axes[0], mae, "MAE", "#2563eb"),
        (axes[1], rmse, "RMSE", "#7c3aed"),
    ):
        bars = axis.bar(x_positions, values, color=color, alpha=0.86)
        axis.set_xticks(x_positions, labels, rotation=35, ha="right")
        axis.set_ylabel(metric)
        axis.set_ylim(bottom=0.0)
        axis.set_title(f"Test overall {metric}")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    figure.suptitle("Unified post-hoc model comparison (UTC test period)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_horizon_rmse(
    model_results: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 5.6))
    for model_name in REQUIRED_MODELS:
        rows = model_results[model_name]["aggregate"]["by_horizon"]
        axis.plot(
            [row["horizon_hour"] for row in rows],
            [row["rmse"]["mean"] for row in rows],
            linewidth=1.7,
            label=MODEL_LABELS[model_name],
        )
    axis.set_xlabel("Forecast horizon hour")
    axis.set_ylabel("RMSE")
    axis.set_title("Test RMSE by forecast horizon (post-hoc descriptive)")
    axis.set_xticks(np.arange(1, len(rows) + 1, max(1, len(rows) // 8)))
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_zone_rmse(
    model_results: dict[str, object],
    output_path: Path,
) -> None:
    zones = list(model_results["ordinary_lstm"]["aggregate"]["by_zone"])
    x_positions = np.arange(len(zones), dtype=np.float64)
    width = 0.11
    figure, axis = plt.subplots(figsize=(11.5, 5.4))
    for model_index, model_name in enumerate(REQUIRED_MODELS):
        values = [
            model_results[model_name]["aggregate"]["by_zone"][zone]["rmse"]["mean"]
            for zone in zones
        ]
        offset = (model_index - (len(REQUIRED_MODELS) - 1) / 2.0) * width
        axis.bar(
            x_positions + offset,
            values,
            width=width,
            label=MODEL_LABELS[model_name],
        )
    axis.set_xticks(x_positions, [f"Zone {zone}" for zone in zones])
    axis.set_ylabel("RMSE")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Test RMSE by zone (post-hoc descriptive)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_analysis(
    config_path: str | Path,
    *,
    predictions_by_model: dict[str, np.ndarray] | None = None,
    artifact_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
    n_resamples: int | None = None,
) -> dict[str, object]:
    """Compute fixed exploratory diagnostics without changing model selection."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol: dict[str, Any] = config["exploratory_analysis"]
    if protocol.get("post_hoc_exploratory") is not True:
        raise ValueError("analysis must be labeled post-hoc exploratory")
    block_lengths = [
        int(value) for value in protocol["bootstrap_block_lengths_days"]
    ]
    if int(protocol["primary_block_length_days"]) not in block_lengths:
        raise ValueError("primary bootstrap block length must be in sensitivity list")
    bootstrap_repetitions = int(
        protocol["bootstrap_resamples"] if n_resamples is None else n_resamples
    )
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap resamples must be positive")

    processed_dir = _resolve_path(project_root, config["data"]["processed_dir"])
    metadata = json.loads(
        (processed_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("time_standard") != "UTC":
        raise ValueError("processed data must declare UTC as its time standard")
    train_split = _load_split(processed_dir / "train.npz")
    test_split = _load_split(processed_dir / "test.npz")
    mask_result = training_derived_masks(
        train_split,
        test_split,
        high_power_quantile=float(protocol["high_power_quantile"]),
        ramp_quantile=float(protocol["ramp_quantile"]),
    )
    irradiance_index = list(metadata["nwp_feature_names"]).index("var169_hourly")
    masks = dict(mask_result["masks"])
    masks["irradiance_active"] = (
        test_split["future_nwp_raw"][:, :, irradiance_index] > 0.0
    )

    if predictions_by_model is None:
        predictions_by_model = _load_saved_predictions(
            project_root,
            config,
            metadata,
            test_split,
        )
    missing_models = sorted(set(REQUIRED_MODELS).difference(predictions_by_model))
    if missing_models:
        raise ValueError(f"missing required model predictions: {missing_models}")
    extra_models = sorted(set(predictions_by_model).difference(REQUIRED_MODELS))
    if extra_models:
        raise ValueError(f"unexpected model predictions: {extra_models}")

    model_results = {
        model_name: aggregate_prediction_diagnostics(
            test_split["target_power"],
            predictions_by_model[model_name],
            test_split["zone_id"],
            masks,
        )
        for model_name in REQUIRED_MODELS
    }
    paired_results: dict[str, object] = {}
    for comparison_name, (candidate_name, reference_name) in COMPARISONS.items():
        paired_results[comparison_name] = {}
        for metric in ("mae", "rmse"):
            paired_results[comparison_name][metric] = {
                str(block_length): paired_moving_block_bootstrap(
                    test_split["target_power"],
                    predictions_by_model[candidate_name],
                    predictions_by_model[reference_name],
                    test_split["origin_timestamp_utc"],
                    test_split["zone_id"],
                    block_length_days=block_length,
                    n_resamples=bootstrap_repetitions,
                    random_seed=int(protocol["bootstrap_seed"]),
                    metric=metric,
                )
                for block_length in block_lengths
            }

    artifact_dir = _resolve_path(
        project_root,
        artifact_dir if artifact_dir is not None else "artifacts/exploratory_analysis",
    )
    figure_dir = _resolve_path(
        project_root,
        figure_dir if figure_dir is not None else "reports/figures",
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "dataset": metadata["dataset"],
        "time_standard": "UTC",
        "analysis_status": "post_hoc_exploratory",
        "main_model_reselected": False,
        "interpretation_limits": [
            "Test results were already observed before this analysis.",
            "Subgroup results are descriptive and do not show advance event identification.",
            "A bootstrap interval crossing zero does not establish model equivalence.",
            "Bootstrap quantifies test-window uncertainty; seed standard deviations quantify training randomness.",
        ],
        "thresholds": mask_result["thresholds"],
        "mask_counts": {
            name: int(np.asarray(mask, dtype=bool).sum())
            for name, mask in masks.items()
        },
        "bootstrap_protocol": {
            "sampling_unit": "consecutive UTC forecast-origin date blocks",
            "all_zones_and_24_horizons_retained": True,
            "paired_model_indices": True,
            "block_lengths_days": block_lengths,
            "primary_block_length_days": int(protocol["primary_block_length_days"]),
            "n_resamples": bootstrap_repetitions,
            "random_seed": int(protocol["bootstrap_seed"]),
            "estimator": "mean of per-seed metrics recomputed on each sampled test window",
        },
        "models": model_results,
        "paired_bootstrap": paired_results,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_unified_comparison(
        model_results,
        figure_dir / "unified_model_comparison.png",
    )
    _plot_horizon_rmse(
        model_results,
        figure_dir / "model_horizon_rmse.png",
    )
    _plot_zone_rmse(
        model_results,
        figure_dir / "model_zone_rmse.png",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/day_ahead.yaml",
        help="Path to the fixed experiment YAML configuration.",
    )
    args = parser.parse_args()
    results = run_analysis(args.config)
    primary = str(results["bootstrap_protocol"]["primary_block_length_days"])
    print("Post-hoc exploratory analysis complete; main model was not reselected.")
    for comparison_name, comparison in results["paired_bootstrap"].items():
        record = comparison["rmse"][primary]
        print(
            f"{comparison_name}: RMSE difference={record['point_estimate']:.6f}, "
            f"95% block-bootstrap interval="
            f"[{record['confidence_interval_95'][0]:.6f}, "
            f"{record['confidence_interval_95'][1]:.6f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
