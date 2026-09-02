"""Train fixed post-hoc History-only and NWP-only LSTM ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.lstm_model import (  # noqa: E402
    CovariateOnlyLSTM,
    Seq2SeqLSTM,
    SequenceModelArrays,
    build_sequence_arrays,
)
from renewable_forecasting.lstm_training import (  # noqa: E402
    TrainingHistory,
    fit_lstm,
    predict_lstm,
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


VARIANTS = ("history_only", "nwp_only")


def _model_for_variant(
    variant: str,
    architecture: dict[str, int],
) -> torch.nn.Module:
    if variant == "history_only":
        return Seq2SeqLSTM(**architecture)
    if variant == "nwp_only":
        return CovariateOnlyLSTM(**architecture)
    raise ValueError(f"unsupported LSTM ablation variant: {variant}")


def _variant_feature_design(variant: str) -> str:
    if variant == "history_only":
        return (
            "24-hour historical power encoder; decoder uses four UTC calendar "
            "features and zone embedding; all NWP inputs excluded"
        )
    if variant == "nwp_only":
        return (
            "No historical-power encoder or derived hidden state; zero-state "
            "decoder uses train-scaled future NWP, four UTC calendar features, "
            "and zone embedding"
        )
    raise ValueError(f"unsupported LSTM ablation variant: {variant}")


def _validate_feature_counts(
    arrays_by_split: dict[str, SequenceModelArrays],
) -> int:
    counts = {
        arrays.future_covariates.shape[2]
        for arrays in arrays_by_split.values()
    }
    if len(counts) != 1:
        raise ValueError("train, validation, and test covariate schemas differ")
    return counts.pop()


def run_training(
    config_path: str | Path,
    artifact_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train both fixed ablations with validation-only early stopping."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol: dict[str, Any] = config["lstm"]
    ablation_protocol: dict[str, Any] = config["lstm_ablations"]
    if protocol["loss"] != "mse":
        raise ValueError("LSTM ablation training requires MSE loss")
    if ablation_protocol.get("post_hoc_exploratory") is not True:
        raise ValueError("LSTM ablations must be labeled post-hoc exploratory")
    variants = [str(value) for value in ablation_protocol["variants"]]
    if variants != list(VARIANTS):
        raise ValueError(f"fixed LSTM ablation variants must be {list(VARIANTS)}")

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

    full_control_path = _resolve_path(
        project_root,
        ablation_protocol["full_input_control_metrics"],
    )
    if not full_control_path.is_file():
        raise FileNotFoundError(f"full-input LSTM metrics not found: {full_control_path}")
    full_control = json.loads(full_control_path.read_text(encoding="utf-8"))

    artifact_dir = _resolve_path(
        project_root,
        artifact_dir if artifact_dir is not None else "artifacts/lstm_ablations",
    )
    checkpoint_dir = _resolve_path(
        project_root,
        checkpoint_dir if checkpoint_dir is not None else "artifacts/checkpoints",
    )
    figure_dir = _resolve_path(
        project_root,
        figure_dir if figure_dir is not None else "reports/figures",
    )
    for directory in (artifact_dir, checkpoint_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    splits = {
        name: _load_split(processed_dir / f"{name}.npz")
        for name in ("train", "validation", "test")
    }
    seeds = [int(seed) for seed in config["evaluation"]["random_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation random seeds must be unique and non-empty")
    device = _resolved_device(str(protocol["device"]))
    clip_bounds = tuple(float(value) for value in protocol["prediction_clip"])

    variant_results: dict[str, object] = {}
    prediction_artifact: dict[str, np.ndarray] = {
        "seeds": np.asarray(seeds, dtype=np.int32)
    }
    for variant in variants:
        arrays_by_split = {
            name: build_sequence_arrays(
                split,
                zone_categories,
                input_variant=variant,
            )
            for name, split in splits.items()
        }
        future_feature_count = _validate_feature_counts(arrays_by_split)
        architecture = {
            "future_feature_count": future_feature_count,
            "zone_count": len(zone_categories),
            "zone_embedding_dim": int(protocol["zone_embedding_dim"]),
            "hidden_size": int(protocol["hidden_size"]),
            "num_layers": int(protocol["num_layers"]),
        }
        count_model = _model_for_variant(variant, architecture)
        parameter_count = sum(
            parameter.numel() for parameter in count_model.parameters()
        )
        del count_model

        runs: list[dict[str, object]] = []
        histories: list[tuple[int, TrainingHistory]] = []
        validation_predictions: list[np.ndarray] = []
        test_predictions: list[np.ndarray] = []
        for seed in seeds:
            set_reproducible_seed(
                seed,
                deterministic_algorithms=bool(protocol["deterministic_algorithms"]),
            )
            model = _model_for_variant(variant, architecture)
            history = fit_lstm(
                model,
                arrays_by_split["train"],
                arrays_by_split["validation"],
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
                arrays_by_split["validation"],
                batch_size=int(protocol["batch_size"]),
                device=device,
                clip_bounds=clip_bounds,
                num_workers=int(protocol["num_workers"]),
            )
            test_prediction = predict_lstm(
                model,
                arrays_by_split["test"],
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
                        splits["validation"], validation_prediction, irradiance_index
                    ),
                    "test": _evaluate_prediction(
                        splits["test"], test_prediction, irradiance_index
                    ),
                }
            )
            histories.append((seed, history))
            validation_predictions.append(validation_prediction)
            test_predictions.append(test_prediction)
            torch.save(
                {
                    "model": f"lstm_{variant}",
                    "input_variant": variant,
                    "seed": seed,
                    "architecture": architecture,
                    "model_state_dict": _portable_state_dict(model),
                    "training_history": asdict(history),
                    "zone_categories": zone_categories,
                    "nwp_feature_names": nwp_feature_names,
                    "calendar_feature_names": calendar_feature_names,
                    "prediction_clip": clip_bounds,
                    "time_standard": "UTC",
                    "analysis_status": "post_hoc_exploratory",
                },
                checkpoint_dir / f"lstm_{variant}_seed_{seed}.pt",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        validation_tensor = np.stack(validation_predictions).astype(np.float32)
        test_tensor = np.stack(test_predictions).astype(np.float32)
        prediction_artifact[f"{variant}_validation"] = validation_tensor
        prediction_artifact[f"{variant}_test"] = test_tensor
        variant_results[variant] = {
            "model": f"lstm_{variant}",
            "feature_design": _variant_feature_design(variant),
            "architecture": architecture,
            "parameter_count": int(parameter_count),
            "selection_rule": {
                "metric": "validation_mse",
                "test_used_for_selection": False,
                "best_checkpoint_restored": True,
                "early_stopping_patience": int(protocol["early_stopping_patience"]),
                "early_stopping_min_delta": float(
                    protocol["early_stopping_min_delta"]
                ),
            },
            "runs": runs,
            "aggregate": aggregate_seed_metrics(runs),
        }
        _plot_training_curves(
            histories,
            figure_dir / f"lstm_{variant}_training_curves.png",
        )

    results: dict[str, object] = {
        "dataset": metadata["dataset"],
        "time_standard": "UTC",
        "analysis_status": "post_hoc_exploratory",
        "main_model_reselected": False,
        "full_input_control": {
            "reused": True,
            "metrics_path": str(ablation_protocol["full_input_control_metrics"]),
            "model": full_control.get("model", "ordinary_seq2seq_lstm"),
            "aggregate": full_control["aggregate"],
        },
        "training_protocol_source": "lstm",
        "training_protocol": {
            key: protocol[key]
            for key in (
                "device", "hidden_size", "num_layers", "zone_embedding_dim",
                "batch_size", "learning_rate", "weight_decay", "max_epochs",
                "early_stopping_patience", "early_stopping_min_delta",
                "gradient_clip_norm", "loss", "prediction_clip",
                "deterministic_algorithms", "num_workers",
            )
        },
        "random_seeds": seeds,
        "row_counts": {
            name: int(split["target_power"].shape[0])
            for name, split in splits.items()
        },
        "variants": variant_results,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(artifact_dir / "predictions.npz", **prediction_artifact)
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
    results = run_training(args.config)
    for variant, result in results["variants"].items():
        overall = result["aggregate"]["test"]["overall"]
        print(
            f"{variant}: test MAE={overall['mae']['mean']:.6f} "
            f"+/- {overall['mae']['std']:.6f}; "
            f"RMSE={overall['rmse']['mean']:.6f} "
            f"+/- {overall['rmse']['std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
