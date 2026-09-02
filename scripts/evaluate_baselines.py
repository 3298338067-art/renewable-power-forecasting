"""Evaluate leakage-safe persistence baselines on validation and test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.baselines import (  # noqa: E402
    daily_persistence,
    seasonal_persistence,
)
from renewable_forecasting.metrics import summarize_forecast_errors  # noqa: E402


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"processed split not found: {path}")
    with np.load(path) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _plot_forecast_example(
    split: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    index = 0
    timestamps = pd.to_datetime(
        split["target_timestamp_utc"][index], unit="ns", utc=True
    )
    zone_id = int(split["zone_id"][index])

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        timestamps,
        split["target_power"][index],
        color="#111827",
        linewidth=2.2,
        label="Observed",
    )
    axis.plot(
        timestamps,
        predictions["daily_persistence"][index],
        color="#2563eb",
        linewidth=1.8,
        label="Daily persistence",
    )
    axis.plot(
        timestamps,
        predictions["seasonal_persistence_7d"][index],
        color="#dc2626",
        linewidth=1.8,
        label="7-day persistence",
    )
    axis.set_title(
        f"Zone {zone_id} day-ahead baseline forecast (UTC): "
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


def _plot_horizon_rmse(
    test_results: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    styles = {
        "daily_persistence": ("#2563eb", "Daily persistence"),
        "seasonal_persistence_7d": ("#dc2626", "7-day persistence"),
    }
    for baseline_name, (color, label) in styles.items():
        horizon_rows = test_results[baseline_name]["by_horizon"]
        axis.plot(
            [row["horizon_hour"] for row in horizon_rows],
            [row["rmse"] for row in horizon_rows],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
            label=label,
        )

    axis.set_xticks(np.arange(1, 25))
    axis.set_xlabel("Forecast horizon after 00:00 UTC issue (hour)")
    axis.set_ylabel("RMSE")
    axis.set_title("Test-set RMSE by UTC forecast horizon")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_evaluation(
    config_path: str | Path,
    artifact_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate both baselines and persist metrics plus diagnostic figures."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_config = config["data"]
    processed_dir = _resolve_path(project_root, data_config["processed_dir"])
    interim_path = _resolve_path(project_root, data_config["interim_csv"])
    if not interim_path.is_file():
        raise FileNotFoundError(f"interim hourly panel not found: {interim_path}")
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"processed metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nwp_feature_names = metadata["nwp_feature_names"]
    try:
        irradiance_index = nwp_feature_names.index("var169_hourly")
    except ValueError as error:
        raise ValueError("processed data must include var169_hourly") from error

    artifact_dir = _resolve_path(
        project_root,
        artifact_dir
        if artifact_dir is not None
        else Path("artifacts") / "baselines",
    )
    figure_dir = _resolve_path(
        project_root,
        figure_dir
        if figure_dir is not None
        else Path("reports") / "figures",
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(
        interim_path,
        usecols=["zone_id", "timestamp", "power"],
        parse_dates=["timestamp"],
    )
    split_names = ["validation", "test"]
    results: dict[str, object] = {
        "dataset": "GEFCom2014 Solar Track",
        "time_standard": "UTC",
        "evaluated_splits": split_names,
        "irradiance_active_definition": (
            "forecast VAR169 hourly increment > 0 J m^-2"
        ),
        "baseline_definitions": {
            "daily_persistence": "target power from exactly 24 hours earlier",
            "seasonal_persistence_7d": (
                "target power from the same zone and hour exactly 7 days earlier"
            ),
        },
    }
    saved_splits: dict[str, dict[str, np.ndarray]] = {}
    saved_predictions: dict[str, dict[str, np.ndarray]] = {}

    for split_name in split_names:
        split = _load_split(processed_dir / f"{split_name}.npz")
        predictions = {
            "daily_persistence": daily_persistence(split["history_power"]),
            "seasonal_persistence_7d": seasonal_persistence(
                panel,
                split["zone_id"],
                split["target_timestamp_utc"],
            ),
        }
        irradiance_active_mask = (
            split["future_nwp_raw"][:, :, irradiance_index] > 0.0
        )
        results[split_name] = {
            baseline_name: summarize_forecast_errors(
                split["target_power"],
                forecast,
                split["zone_id"],
                irradiance_active_mask=irradiance_active_mask,
            )
            for baseline_name, forecast in predictions.items()
        }
        saved_splits[split_name] = split
        saved_predictions[split_name] = predictions

    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_forecast_example(
        saved_splits["test"],
        saved_predictions["test"],
        figure_dir / "baseline_forecast_example.png",
    )
    _plot_horizon_rmse(
        results["test"],
        figure_dir / "baseline_horizon_rmse.png",
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
    results = run_evaluation(args.config)
    print("Evaluated day-ahead persistence baselines")
    for split_name in results["evaluated_splits"]:
        print(f"  {split_name}:")
        for baseline_name, summary in results[split_name].items():
            overall = summary["overall"]
            print(
                f"    {baseline_name}: "
                f"MAE={overall['mae']:.6f}, RMSE={overall['rmse']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
