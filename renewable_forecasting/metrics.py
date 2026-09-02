"""Forecast-error metrics with JSON-ready grouped summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _validated_pair(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted arrays must have identical shapes")
    if actual_values.size == 0:
        raise ValueError("forecast arrays must not be empty")
    if not np.isfinite(actual_values).all() or not np.isfinite(
        predicted_values
    ).all():
        raise ValueError("forecast arrays must contain only finite values")
    return actual_values, predicted_values


def compute_error_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    """Return MAE and RMSE over every supplied forecast value."""
    actual_values, predicted_values = _validated_pair(actual, predicted)
    errors = predicted_values - actual_values
    rmse = float(np.sqrt(np.mean(np.square(errors), dtype=np.float64)))
    return {
        "count": int(errors.size),
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": rmse,
        "nrmse_capacity_percent": 100.0 * rmse,
    }


def summarize_forecast_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
    zone_ids: np.ndarray,
    irradiance_active_mask: np.ndarray | None = None,
) -> Mapping[str, object]:
    """Summarize errors overall, by zone, and by forecast horizon."""
    actual_values, predicted_values = _validated_pair(actual, predicted)
    if actual_values.ndim != 2:
        raise ValueError("forecast arrays must have shape [samples, horizon]")

    zones = np.asarray(zone_ids)
    if zones.ndim != 1 or zones.size != actual_values.shape[0]:
        raise ValueError("zone IDs must contain one value per forecast sample")

    by_zone = {
        str(int(zone)): compute_error_metrics(
            actual_values[zones == zone],
            predicted_values[zones == zone],
        )
        for zone in np.unique(zones)
    }
    by_horizon = [
        {
            "horizon_hour": horizon_index + 1,
            **compute_error_metrics(
                actual_values[:, horizon_index],
                predicted_values[:, horizon_index],
            ),
        }
        for horizon_index in range(actual_values.shape[1])
    ]
    summary: dict[str, object] = {
        "overall": compute_error_metrics(actual_values, predicted_values),
        "by_zone": by_zone,
        "by_horizon": by_horizon,
    }
    if irradiance_active_mask is not None:
        mask = np.asarray(irradiance_active_mask, dtype=bool)
        if mask.shape != actual_values.shape:
            raise ValueError(
                "irradiance-active mask must match forecast array shape"
            )
        if not mask.any():
            raise ValueError("irradiance-active mask must select at least one value")
        summary["irradiance_active"] = compute_error_metrics(
            actual_values[mask],
            predicted_values[mask],
        )
    return summary
