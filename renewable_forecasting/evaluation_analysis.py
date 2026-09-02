"""Pure helpers for exploratory forecast diagnostics and paired uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from renewable_forecasting.metrics import (
    compute_error_metrics,
    summarize_forecast_errors,
)


DAY_NS = 86_400_000_000_000
METRIC_NAMES = ("mae", "rmse", "nrmse_capacity_percent")


def _forecast_pair(split: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "history_power" not in split or "target_power" not in split:
        raise ValueError("split must contain history_power and target_power")
    history = np.asarray(split["history_power"], dtype=np.float64)
    target = np.asarray(split["target_power"], dtype=np.float64)
    if history.ndim != 3 or history.shape[2] != 1:
        raise ValueError("history_power must have shape [samples, history, 1]")
    if target.ndim != 2 or target.shape[0] != history.shape[0]:
        raise ValueError("target_power must have shape [samples, horizon]")
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError("power arrays must contain only finite values")
    return history, target


def _ramp_magnitudes(
    history_power: np.ndarray,
    target_power: np.ndarray,
) -> np.ndarray:
    preceding = np.concatenate(
        [history_power[:, -1, 0:1], target_power[:, :-1]],
        axis=1,
    )
    return np.abs(target_power - preceding)


def training_derived_masks(
    train_split: Mapping[str, np.ndarray],
    evaluated_split: Mapping[str, np.ndarray],
    *,
    high_power_quantile: float = 0.75,
    ramp_quantile: float = 0.90,
) -> dict[str, object]:
    """Create post-hoc masks using thresholds derived only from training data."""
    for label, value in (
        ("high_power_quantile", high_power_quantile),
        ("ramp_quantile", ramp_quantile),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{label} must be strictly between zero and one")
    train_history, train_target = _forecast_pair(train_split)
    evaluated_history, evaluated_target = _forecast_pair(evaluated_split)
    positive_train_power = train_target[train_target > 0.0]
    if positive_train_power.size == 0:
        raise ValueError("training targets must contain positive power values")
    train_ramps = _ramp_magnitudes(train_history, train_target)
    high_power_threshold = float(
        np.quantile(positive_train_power, high_power_quantile)
    )
    ramp_threshold = float(np.quantile(train_ramps, ramp_quantile))
    evaluated_ramps = _ramp_magnitudes(evaluated_history, evaluated_target)
    return {
        "thresholds": {
            "high_power": high_power_threshold,
            "high_power_quantile": float(high_power_quantile),
            "ramp_magnitude": ramp_threshold,
            "ramp_quantile": float(ramp_quantile),
            "derived_from": "training_target_power",
        },
        "masks": {
            "high_power": evaluated_target >= high_power_threshold,
            "ramp": evaluated_ramps >= ramp_threshold,
        },
    }


def _validated_prediction_tensor(
    actual: np.ndarray,
    predictions_by_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predictions = np.asarray(predictions_by_seed, dtype=np.float64)
    if actual_values.ndim != 2:
        raise ValueError("actual forecasts must have shape [samples, horizon]")
    if predictions.ndim != 3 or predictions.shape[1:] != actual_values.shape:
        raise ValueError(
            "predictions must have shape [seeds, samples, horizon]"
        )
    if predictions.shape[0] == 0:
        raise ValueError("predictions must contain at least one seed")
    if not np.isfinite(actual_values).all() or not np.isfinite(predictions).all():
        raise ValueError("actual and prediction arrays must contain finite values")
    return actual_values, predictions


def _aggregate_metric_records(
    records: list[Mapping[str, float | int]],
) -> dict[str, object]:
    if not records:
        raise ValueError("metric records must not be empty")
    result: dict[str, object] = {"count": int(records[0]["count"])}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [float(record[metric_name]) for record in records],
            dtype=np.float64,
        )
        result[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return result


def aggregate_prediction_diagnostics(
    actual: np.ndarray,
    predictions_by_seed: np.ndarray,
    zone_ids: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Aggregate overall and grouped metrics without averaging predictions."""
    actual_values, predictions = _validated_prediction_tensor(
        actual,
        predictions_by_seed,
    )
    zones = np.asarray(zone_ids)
    if zones.ndim != 1 or zones.size != actual_values.shape[0]:
        raise ValueError("zone IDs must contain one value per forecast sample")
    validated_masks: dict[str, np.ndarray] = {}
    for mask_name, mask in masks.items():
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != actual_values.shape:
            raise ValueError(f"{mask_name} mask must match forecast shape")
        if not mask_values.any():
            raise ValueError(f"{mask_name} mask must select at least one value")
        validated_masks[mask_name] = mask_values

    runs: list[dict[str, object]] = []
    for seed_prediction in predictions:
        summary = dict(
            summarize_forecast_errors(
                actual_values,
                seed_prediction,
                zones,
            )
        )
        for mask_name, mask in validated_masks.items():
            summary[mask_name] = compute_error_metrics(
                actual_values[mask],
                seed_prediction[mask],
            )
        runs.append(summary)

    aggregate: dict[str, object] = {
        "overall": _aggregate_metric_records(
            [run["overall"] for run in runs]
        ),
        "by_zone": {
            zone: _aggregate_metric_records(
                [run["by_zone"][zone] for run in runs]
            )
            for zone in runs[0]["by_zone"]
        },
        "by_horizon": [
            {
                "horizon_hour": horizon_index + 1,
                **_aggregate_metric_records(
                    [run["by_horizon"][horizon_index] for run in runs]
                ),
            }
            for horizon_index in range(actual_values.shape[1])
        ],
    }
    for mask_name in validated_masks:
        aggregate[mask_name] = _aggregate_metric_records(
            [run[mask_name] for run in runs]
        )
    return {
        "seed_count": int(predictions.shape[0]),
        "runs": runs,
        "aggregate": aggregate,
    }


def _validated_origins_and_zones(
    origin_timestamps: np.ndarray,
    zone_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    origins = np.asarray(origin_timestamps).astype("datetime64[ns]").astype(np.int64)
    zones = np.asarray(zone_ids)
    if origins.ndim != 1 or zones.ndim != 1 or origins.size != zones.size:
        raise ValueError("origin timestamps and zone IDs must be aligned vectors")
    unique_origins = np.unique(origins)
    if unique_origins.size == 0:
        raise ValueError("origin timestamps must not be empty")
    if unique_origins.size > 1 and not np.all(np.diff(unique_origins) == DAY_NS):
        raise ValueError("origin timestamps must cover consecutive UTC dates")
    expected_zones = set(zones.tolist())
    groups: list[np.ndarray] = []
    for origin in unique_origins:
        rows = np.flatnonzero(origins == origin)
        group_zones = zones[rows]
        if (
            rows.size != len(expected_zones)
            or set(group_zones.tolist()) != expected_zones
            or np.unique(group_zones).size != rows.size
        ):
            raise ValueError("every origin date must contain the same zone set once")
        groups.append(rows)
    return unique_origins, zones, groups


def moving_block_origin_indices(
    origin_timestamps: np.ndarray,
    zone_ids: np.ndarray,
    *,
    block_length_days: int,
    n_resamples: int,
    random_seed: int,
) -> np.ndarray:
    """Draw non-circular consecutive-date block indices."""
    unique_origins, _, _ = _validated_origins_and_zones(
        origin_timestamps,
        zone_ids,
    )
    date_count = unique_origins.size
    if not 1 <= block_length_days <= date_count:
        raise ValueError("block length must be between one and the date count")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    block_count = math.ceil(date_count / block_length_days)
    maximum_start = date_count - block_length_days
    generator = np.random.default_rng(random_seed)
    sampled = np.empty((n_resamples, date_count), dtype=np.int64)
    offsets = np.arange(block_length_days, dtype=np.int64)
    for replicate_index in range(n_resamples):
        starts = generator.integers(
            0,
            maximum_start + 1,
            size=block_count,
        )
        sampled[replicate_index] = np.concatenate(
            [start + offsets for start in starts]
        )[:date_count]
    return sampled


def _metric_value(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: str,
) -> float:
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual,
        dtype=np.float64,
    )
    if metric == "mae":
        return float(np.mean(np.abs(errors), dtype=np.float64))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(errors), dtype=np.float64)))
    raise ValueError("metric must be 'mae' or 'rmse'")


def paired_moving_block_bootstrap(
    actual: np.ndarray,
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    origin_timestamps: np.ndarray,
    zone_ids: np.ndarray,
    *,
    block_length_days: int,
    n_resamples: int,
    random_seed: int,
    metric: str,
) -> dict[str, object]:
    """Estimate candidate-reference metric differences with paired date blocks."""
    actual_values, candidate = _validated_prediction_tensor(
        actual,
        candidate_predictions,
    )
    _, reference = _validated_prediction_tensor(actual, reference_predictions)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference predictions must have equal shapes")
    unique_origins, _, groups = _validated_origins_and_zones(
        origin_timestamps,
        zone_ids,
    )
    if unique_origins.size != len(groups) or actual_values.shape[0] != sum(
        group.size for group in groups
    ):
        raise ValueError("forecast rows do not align with origin-date groups")
    if metric not in {"mae", "rmse"}:
        raise ValueError("metric must be 'mae' or 'rmse'")

    candidate_point = float(
        np.mean(
            [_metric_value(actual_values, values, metric) for values in candidate]
        )
    )
    reference_point = float(
        np.mean(
            [_metric_value(actual_values, values, metric) for values in reference]
        )
    )
    sampled_origins = moving_block_origin_indices(
        origin_timestamps,
        zone_ids,
        block_length_days=block_length_days,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    def origin_error_sums(predictions: np.ndarray) -> np.ndarray:
        errors = predictions - actual_values[None, ...]
        contributions = np.abs(errors) if metric == "mae" else np.square(errors)
        return np.stack(
            [contributions[:, rows, :].sum(axis=(1, 2)) for rows in groups],
            axis=1,
        )

    sampled_value_count = (
        sampled_origins.shape[1] * groups[0].size * actual_values.shape[1]
    )
    candidate_sums = origin_error_sums(candidate)[:, sampled_origins].sum(axis=2)
    reference_sums = origin_error_sums(reference)[:, sampled_origins].sum(axis=2)
    candidate_replicates = candidate_sums / sampled_value_count
    reference_replicates = reference_sums / sampled_value_count
    if metric == "rmse":
        candidate_replicates = np.sqrt(candidate_replicates)
        reference_replicates = np.sqrt(reference_replicates)
    differences = candidate_replicates.mean(axis=0) - reference_replicates.mean(
        axis=0
    )

    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "metric": metric,
        "difference": "candidate_minus_reference",
        "negative_favors": "candidate",
        "candidate_metric": candidate_point,
        "reference_metric": reference_point,
        "point_estimate": candidate_point - reference_point,
        "confidence_interval_95": [float(lower), float(upper)],
        "interval_crosses_zero": bool(lower <= 0.0 <= upper),
        "block_length_days": int(block_length_days),
        "date_count": int(unique_origins.size),
        "n_resamples": int(n_resamples),
        "random_seed": int(random_seed),
    }
