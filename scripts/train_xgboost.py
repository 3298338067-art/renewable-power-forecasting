"""Train and evaluate a leakage-safe XGBoost day-ahead solar forecast."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.baselines import daily_persistence  # noqa: E402
from renewable_forecasting.metrics import summarize_forecast_errors  # noqa: E402
from renewable_forecasting.xgboost_model import (  # noqa: E402
    LongHorizonTable,
    aggregate_seed_metrics,
    build_long_horizon_table,
    clip_normalized_power,
    reshape_long_predictions,
    select_best_candidate,
)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"processed split not found: {path}")
    with np.load(path) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _make_model(
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
) -> xgb.XGBRegressor:
    candidate_parameters = {
        key: value for key, value in candidate.items() if key != "name"
    }
    return xgb.XGBRegressor(
        **candidate_parameters,
        objective=protocol["objective"],
        eval_metric=protocol["eval_metric"],
        tree_method=protocol["tree_method"],
        device=protocol["device"],
        early_stopping_rounds=protocol["early_stopping_rounds"],
        n_jobs=protocol["n_jobs"],
        random_state=seed,
        verbosity=0,
    )


def _fit_model(
    model: xgb.XGBRegressor,
    train_table: LongHorizonTable,
    validation_table: LongHorizonTable,
) -> None:
    model.fit(
        train_table.features,
        train_table.targets,
        eval_set=[(validation_table.features, validation_table.targets)],
        verbose=False,
    )


def _predict_table(
    model: xgb.XGBRegressor,
    table: LongHorizonTable,
    clip_bounds: tuple[float, float],
) -> np.ndarray:
    flat_predictions = model.predict(table.features)
    predictions = reshape_long_predictions(
        flat_predictions,
        sample_count=table.sample_count,
        horizon_hours=table.horizon_hours,
    )
    return clip_normalized_power(predictions, *clip_bounds)


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


def _best_iteration(model: xgb.XGBRegressor) -> int | None:
    try:
        return int(model.best_iteration)
    except (AttributeError, TypeError):
        return None


def _plot_feature_importance(
    feature_names: tuple[str, ...],
    importance: np.ndarray,
    output_path: Path,
) -> None:
    top_count = min(20, len(feature_names))
    top_indices = np.argsort(importance)[-top_count:]
    figure, axis = plt.subplots(figsize=(9, 6.8))
    axis.barh(
        np.asarray(feature_names)[top_indices],
        importance[top_indices],
        color="#2563eb",
    )
    axis.set_xlabel("Mean XGBoost feature importance across seeds")
    axis.set_title("Top XGBoost features (descriptive, not causal)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_forecast_comparison(
    test_split: dict[str, np.ndarray],
    xgboost_prediction: np.ndarray,
    output_path: Path,
) -> None:
    sample_index = 0
    timestamps = pd.to_datetime(
        test_split["target_timestamp_utc"][sample_index],
        unit="ns",
        utc=True,
    )
    persistence = daily_persistence(test_split["history_power"])
    figure, axis = plt.subplots(figsize=(10, 4.8))
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
        linewidth=1.6,
        label="Daily persistence",
    )
    axis.plot(
        timestamps,
        xgboost_prediction[sample_index],
        color="#2563eb",
        linewidth=1.9,
        label="XGBoost mean",
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
    figure_dir: str | Path | None = None,
) -> dict[str, object]:
    """Select on validation data, then evaluate the fixed model on test data."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    protocol = config["xgboost"]
    processed_dir = _resolve_path(
        project_root,
        config["data"]["processed_dir"],
    )
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"processed metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("time_standard") != "UTC":
        raise ValueError("processed data must declare UTC as its time standard")

    nwp_feature_names = metadata["nwp_feature_names"]
    calendar_feature_names = metadata["calendar_feature_names"]
    zone_categories = metadata["zones"]
    try:
        irradiance_index = nwp_feature_names.index("var169_hourly")
    except ValueError as error:
        raise ValueError("processed data must include var169_hourly") from error

    artifact_dir = _resolve_path(
        project_root,
        artifact_dir
        if artifact_dir is not None
        else Path("artifacts") / "xgboost",
    )
    figure_dir = _resolve_path(
        project_root,
        figure_dir
        if figure_dir is not None
        else Path("reports") / "figures",
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_split = _load_split(processed_dir / "train.npz")
    validation_split = _load_split(processed_dir / "validation.npz")
    train_table = build_long_horizon_table(
        train_split,
        nwp_feature_names,
        calendar_feature_names,
        zone_categories,
    )
    validation_table = build_long_horizon_table(
        validation_split,
        nwp_feature_names,
        calendar_feature_names,
        zone_categories,
    )
    if train_table.feature_names != validation_table.feature_names:
        raise ValueError("train and validation feature schemas differ")

    clip_values = protocol["prediction_clip"]
    clip_bounds = (float(clip_values[0]), float(clip_values[1]))
    selection_seed = int(protocol["selection_seed"])
    selection_results: list[dict[str, object]] = []
    for candidate in protocol["candidates"]:
        model = _make_model(protocol, candidate, selection_seed)
        _fit_model(model, train_table, validation_table)
        validation_prediction = _predict_table(
            model,
            validation_table,
            clip_bounds,
        )
        validation_metrics = _evaluate_prediction(
            validation_split,
            validation_prediction,
            irradiance_index,
        )
        selection_results.append(
            {
                "name": candidate["name"],
                "validation_rmse": float(
                    validation_metrics["overall"]["rmse"]
                ),
                "validation_mae": float(
                    validation_metrics["overall"]["mae"]
                ),
                "best_iteration": _best_iteration(model),
            }
        )
        del model
        gc.collect()

    selected_record = select_best_candidate(selection_results)
    selected_name = str(selected_record["name"])
    selected_candidate = next(
        candidate
        for candidate in protocol["candidates"]
        if candidate["name"] == selected_name
    )

    test_split = _load_split(processed_dir / "test.npz")
    test_table = build_long_horizon_table(
        test_split,
        nwp_feature_names,
        calendar_feature_names,
        zone_categories,
    )
    if train_table.feature_names != test_table.feature_names:
        raise ValueError("train and test feature schemas differ")

    seeds = [int(seed) for seed in config["evaluation"]["random_seeds"]]
    runs: list[dict[str, object]] = []
    validation_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    feature_importances: list[np.ndarray] = []
    for seed in seeds:
        model = _make_model(protocol, selected_candidate, seed)
        _fit_model(model, train_table, validation_table)
        validation_prediction = _predict_table(
            model,
            validation_table,
            clip_bounds,
        )
        test_prediction = _predict_table(
            model,
            test_table,
            clip_bounds,
        )
        run = {
            "seed": seed,
            "best_iteration": _best_iteration(model),
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
        runs.append(run)
        validation_predictions.append(validation_prediction)
        test_predictions.append(test_prediction)
        feature_importances.append(
            np.asarray(model.feature_importances_, dtype=np.float64)
        )
        model.save_model(str(artifact_dir / f"model_seed_{seed}.json"))
        del model
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
    mean_feature_importance = np.mean(
        np.stack(feature_importances),
        axis=0,
    )

    baseline_path = project_root / "artifacts" / "baselines" / "metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline metrics not found: {baseline_path}")
    baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_comparison: dict[str, object] = {}
    for split_name in ("validation", "test"):
        baseline_summary = baseline_results[split_name]["daily_persistence"]
        baseline_comparison[split_name] = {}
        for scope_name in ("overall", "irradiance_active"):
            baseline_rmse = float(baseline_summary[scope_name]["rmse"])
            model_rmse = float(
                aggregate[split_name][scope_name]["rmse"]["mean"]
            )
            baseline_comparison[split_name][scope_name] = {
                "daily_persistence_rmse": baseline_rmse,
                "xgboost_mean_rmse": model_rmse,
                "rmse_reduction_percent": (
                    100.0 * (baseline_rmse - model_rmse) / baseline_rmse
                ),
            }

    ranked_importance = sorted(
        (
            {
                "feature": feature_name,
                "mean_importance": float(importance),
            }
            for feature_name, importance in zip(
                train_table.feature_names,
                mean_feature_importance,
                strict=True,
            )
        ),
        key=lambda row: row["mean_importance"],
        reverse=True,
    )
    results: dict[str, object] = {
        "dataset": metadata["dataset"],
        "time_standard": "UTC",
        "model": "XGBoost",
        "feature_design": (
            "24 historical power values, target-hour NWP, four UTC calendar "
            "features, forecast horizon hour, and zone one-hot encoding"
        ),
        "feature_count": len(train_table.feature_names),
        "feature_names": list(train_table.feature_names),
        "row_counts": {
            "train": int(train_table.features.shape[0]),
            "validation": int(validation_table.features.shape[0]),
            "test": int(test_table.features.shape[0]),
        },
        "selection_rule": {
            "metric": protocol["selection_metric"],
            "seed": selection_seed,
            "test_used_for_selection": False,
        },
        "selection_results": selection_results,
        "selected_candidate": dict(selected_candidate),
        "prediction_clip": list(clip_bounds),
        "random_seeds": seeds,
        "runs": runs,
        "aggregate": aggregate,
        "baseline_comparison": baseline_comparison,
        "feature_importance": ranked_importance,
        "interpretation_note": (
            "Feature importance is descriptive and does not establish causality."
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
    _plot_feature_importance(
        train_table.feature_names,
        mean_feature_importance,
        figure_dir / "xgboost_feature_importance.png",
    )
    _plot_forecast_comparison(
        test_split,
        mean_test_prediction,
        figure_dir / "xgboost_vs_baseline_forecast.png",
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
    print(f"Selected candidate: {results['selected_candidate']['name']}")
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
