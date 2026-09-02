"""Leakage-safe feature construction for the XGBoost forecasting baseline."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LongHorizonTable:
    """One row per sample and forecast horizon."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    sample_count: int
    horizon_hours: int


def _require_shape(
    name: str,
    values: np.ndarray,
    expected_tail: tuple[int, ...],
    sample_count: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != len(expected_tail) + 1 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"{name} must have shape [samples, {', '.join(map(str, expected_tail))}]"
        )
    if sample_count is not None and array.shape[0] != sample_count:
        raise ValueError(f"{name} must contain one row per forecast sample")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def build_long_horizon_table(
    split: Mapping[str, np.ndarray],
    nwp_feature_names: Sequence[str],
    calendar_feature_names: Sequence[str],
    zone_categories: Sequence[int],
) -> LongHorizonTable:
    """Flatten 24-hour samples without including future target power as input."""
    required = {
        "history_power",
        "future_nwp_raw",
        "future_calendar",
        "target_power",
        "zone_id",
    }
    missing = sorted(required.difference(split))
    if missing:
        raise ValueError(f"processed split is missing arrays: {missing}")

    history = np.asarray(split["history_power"])
    if history.ndim != 3 or history.shape[2] != 1:
        raise ValueError("history_power must have shape [samples, horizon, 1]")
    sample_count, history_hours, _ = history.shape

    future_nwp = np.asarray(split["future_nwp_raw"])
    if future_nwp.ndim != 3 or future_nwp.shape[0] != sample_count:
        raise ValueError(
            "future_nwp_raw must have shape [samples, horizon, features]"
        )
    horizon_hours = future_nwp.shape[1]
    if future_nwp.shape[2] != len(nwp_feature_names):
        raise ValueError("NWP feature names do not match processed feature count")

    calendar = np.asarray(split["future_calendar"])
    if (
        calendar.ndim != 3
        or calendar.shape[:2] != (sample_count, horizon_hours)
        or calendar.shape[2] != len(calendar_feature_names)
    ):
        raise ValueError(
            "future_calendar shape does not match samples, horizon, or names"
        )
    targets = _require_shape(
        "target_power",
        split["target_power"],
        (horizon_hours,),
        sample_count=sample_count,
    )
    zones = np.asarray(split["zone_id"])
    if zones.ndim != 1 or zones.size != sample_count:
        raise ValueError("zone_id must contain one value per forecast sample")

    categories = tuple(int(zone) for zone in zone_categories)
    if len(categories) != len(set(categories)) or not categories:
        raise ValueError("zone categories must be unique and non-empty")
    unknown = sorted(set(int(zone) for zone in np.unique(zones)).difference(categories))
    if unknown:
        raise ValueError(f"unknown zone IDs in processed split: {unknown}")

    for name, values in (
        ("history_power", history),
        ("future_nwp_raw", future_nwp),
        ("future_calendar", calendar),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    row_count = sample_count * horizon_hours
    repeated_history = np.repeat(history[:, :, 0], horizon_hours, axis=0)
    horizon_feature = np.tile(
        np.arange(1, horizon_hours + 1, dtype=np.float32), sample_count
    )[:, None]
    repeated_zones = np.repeat(zones, horizon_hours)
    zone_one_hot = np.column_stack(
        [(repeated_zones == zone).astype(np.float32) for zone in categories]
    )

    features = np.column_stack(
        [
            repeated_history,
            future_nwp.reshape(row_count, -1),
            calendar.reshape(row_count, -1),
            horizon_feature,
            zone_one_hot,
        ]
    ).astype(np.float32)
    feature_names = (
        tuple(
            f"power_lag_{offset}h"
            for offset in range(history_hours - 1, -1, -1)
        )
        + tuple(nwp_feature_names)
        + tuple(calendar_feature_names)
        + ("horizon_hour",)
        + tuple(f"zone_{zone}" for zone in categories)
    )
    if features.shape[1] != len(feature_names):
        raise RuntimeError("constructed feature matrix and names are inconsistent")

    return LongHorizonTable(
        features=features,
        targets=targets.reshape(-1).astype(np.float32),
        feature_names=feature_names,
        sample_count=sample_count,
        horizon_hours=horizon_hours,
    )


def reshape_long_predictions(
    predictions: np.ndarray,
    sample_count: int,
    horizon_hours: int,
) -> np.ndarray:
    """Restore flat row predictions to [samples, horizon]."""
    values = np.asarray(predictions, dtype=np.float32)
    expected = sample_count * horizon_hours
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"predictions must contain exactly {expected} values")
    return values.reshape(sample_count, horizon_hours)


def clip_normalized_power(
    predictions: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Apply predeclared physical bounds to normalized power forecasts."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("prediction bounds must be finite and strictly increasing")
    values = np.asarray(predictions, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    return np.clip(values, lower, upper).astype(np.float32, copy=False)


def select_best_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select the lowest validation RMSE without consulting test metrics."""
    if not candidate_results:
        raise ValueError("at least one candidate result is required")
    for result in candidate_results:
        if "name" not in result or "validation_rmse" not in result:
            raise ValueError("each candidate requires name and validation_rmse")
        if not np.isfinite(float(result["validation_rmse"])):
            raise ValueError("candidate validation RMSE must be finite")
    return min(candidate_results, key=lambda row: float(row["validation_rmse"]))


def aggregate_seed_metrics(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Aggregate key validation and test metrics across random seeds."""
    if not runs:
        raise ValueError("at least one seed run is required")
    metric_names = ("mae", "rmse", "nrmse_capacity_percent")
    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split_name in ("validation", "test"):
        aggregate[split_name] = {}
        for scope_name in ("overall", "irradiance_active"):
            aggregate[split_name][scope_name] = {}
            for metric_name in metric_names:
                try:
                    values = np.asarray(
                        [
                            run[split_name][scope_name][metric_name]
                            for run in runs
                        ],
                        dtype=np.float64,
                    )
                except KeyError:
                    if metric_name == "nrmse_capacity_percent":
                        continue
                    raise
                if not np.isfinite(values).all():
                    raise ValueError("seed metrics must contain only finite values")
                aggregate[split_name][scope_name][metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
    return aggregate
