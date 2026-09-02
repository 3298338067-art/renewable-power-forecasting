"""Build leakage-safe GEFCom2014 day-ahead forecasting artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renewable_forecasting.preprocessing import (  # noqa: E402
    apply_nwp_scaler,
    assign_target_day_splits,
    build_day_ahead_windows,
    fit_nwp_scaler,
    prepare_hourly_panel,
    save_processed_artifacts,
)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _subset_windows(
    windows: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[mask] for name, values in windows.items()}


def _origin_range(origin_timestamps_utc: np.ndarray) -> dict[str, str]:
    origins = pd.to_datetime(origin_timestamps_utc, unit="ns", utc=True)
    return {
        "first": origins.min().isoformat(),
        "last": origins.max().isoformat(),
    }


def run_pipeline(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    interim_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the configured preprocessing pipeline and return saved metadata."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    task = config["task"]
    data_config = config["data"]
    raw_path = _resolve_path(project_root, data_config["raw_csv"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"official source CSV not found: {raw_path}")

    configured_interim = data_config["interim_csv"]
    interim_path = _resolve_path(
        project_root, interim_path if interim_path is not None else configured_interim
    )
    configured_output = data_config["processed_dir"]
    output_dir = _resolve_path(
        project_root, output_dir if output_dir is not None else configured_output
    )

    raw_frame = pd.read_csv(raw_path)
    panel, increment_audit = prepare_hourly_panel(
        raw_frame, data_config["accumulated_nwp"]
    )
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(interim_path, index=False, compression="gzip")

    nwp_feature_names = [
        name.lower() for name in data_config["instantaneous_nwp"]
    ] + [f"{name.lower()}_hourly" for name in data_config["accumulated_nwp"]]
    windows = build_day_ahead_windows(
        panel,
        nwp_columns=nwp_feature_names,
        lookback_hours=int(task["lookback_hours"]),
        horizon_hours=int(task["forecast_horizon_hours"]),
        origin_hour=int(data_config["forecast_origin_hour"]),
    )
    split_masks = assign_target_day_splits(
        windows["origin_timestamp_utc"], data_config["split_dates"]
    )
    assignments = np.sum(np.stack(list(split_masks.values())), axis=0)
    if not np.equal(assignments, 1).all():
        raise ValueError("every constructed window must belong to exactly one split")

    split_windows = {
        split_name: _subset_windows(windows, mask)
        for split_name, mask in split_masks.items()
    }
    scaler = fit_nwp_scaler(
        split_windows["train"]["future_nwp_raw"], nwp_feature_names
    )
    for arrays in split_windows.values():
        arrays["future_nwp_scaled"] = apply_nwp_scaler(
            arrays["future_nwp_raw"], scaler
        )
        for name, values in arrays.items():
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise ValueError(f"non-finite values detected in processed array: {name}")

    zones = sorted(int(zone) for zone in np.unique(windows["zone_id"]))
    sample_counts = {
        split_name: int(arrays["zone_id"].size)
        for split_name, arrays in split_windows.items()
    }
    sample_counts_by_zone = {
        split_name: {
            str(zone): int((arrays["zone_id"] == zone).sum()) for zone in zones
        }
        for split_name, arrays in split_windows.items()
    }
    metadata: dict[str, object] = {
        "dataset": "GEFCom2014 Solar Track",
        "source_csv": data_config["raw_csv"],
        "panel_rows": int(len(panel)),
        "panel_start": panel["timestamp"].min().isoformat(),
        "panel_end": panel["timestamp"].max().isoformat(),
        "zones": zones,
        "lookback_hours": int(task["lookback_hours"]),
        "forecast_horizon_hours": int(task["forecast_horizon_hours"]),
        "forecast_origin_hour": int(data_config["forecast_origin_hour"]),
        "time_standard": "UTC",
        "timestamp_fields": {
            "origin_timestamp_utc": "forecast issue time in UTC",
            "target_timestamp_utc": "forecast valid time in UTC",
        },
        "split_dates": data_config["split_dates"],
        "sample_counts": sample_counts,
        "sample_counts_by_zone": sample_counts_by_zone,
        "origin_ranges": {
            split_name: _origin_range(arrays["origin_timestamp_utc"])
            for split_name, arrays in split_windows.items()
        },
        "nwp_feature_names": nwp_feature_names,
        "calendar_feature_names": [
            "utc_hour_sin",
            "utc_hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ],
        "calendar_time_basis": "UTC",
        "accumulated_increment_audit": increment_audit,
        "negative_increment_policy": "audit_then_clip_to_zero",
        "scaler_fit_scope": data_config["scaler_fit_scope"],
        "power_above_one_count": int((panel["power"] > 1).sum()),
    }
    save_processed_artifacts(output_dir, split_windows, scaler, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "day_ahead.yaml",
        help="Path to the fixed experiment YAML configuration.",
    )
    args = parser.parse_args()
    metadata = run_pipeline(args.config)
    print("Processed GEFCom2014 Solar data")
    print(f"  zones: {metadata['zones']}")
    print(f"  sample counts: {metadata['sample_counts']}")
    print(
        "  accumulated negative increments: "
        + str(
            {
                name: values["negative_count"]
                for name, values in metadata["accumulated_increment_audit"].items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
