"""Leakage-safe baseline forecasts for day-ahead solar power."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_persistence(history_power: np.ndarray) -> np.ndarray:
    """Predict each target hour with the observation exactly 24 hours earlier."""
    history = np.asarray(history_power)
    if history.ndim != 3 or history.shape[1:] != (24, 1):
        raise ValueError("history power must have shape [samples, 24, 1]")
    if not np.isfinite(history).all():
        raise ValueError("history power contains non-finite values")
    return history[:, :, 0].astype(np.float32, copy=True)


def seasonal_persistence(
    panel: pd.DataFrame,
    zone_ids: np.ndarray,
    target_timestamps_utc: np.ndarray,
    lag_days: int = 7,
) -> np.ndarray:
    """Predict targets from the exact same zone and hour several days earlier."""
    required_columns = {"zone_id", "timestamp", "power"}
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise ValueError(f"panel is missing required columns: {missing_columns}")
    if lag_days <= 0:
        raise ValueError("lag_days must be positive")

    targets = np.asarray(target_timestamps_utc, dtype=np.int64)
    zones = np.asarray(zone_ids)
    if targets.ndim != 2:
        raise ValueError("target timestamps must have shape [samples, horizon]")
    if zones.ndim != 1 or zones.size != targets.shape[0]:
        raise ValueError("zone IDs must contain one value per sample")

    source = panel.loc[:, ["zone_id", "timestamp", "power"]].copy()
    source["timestamp"] = pd.to_datetime(
        source["timestamp"], errors="raise", utc=True
    )
    if source.duplicated(["zone_id", "timestamp"]).any():
        raise ValueError("panel contains duplicate zone-timestamp observations")

    lookup = source.set_index(["zone_id", "timestamp"])["power"]
    lag_nanoseconds = pd.Timedelta(days=lag_days).value
    lagged_timestamps = pd.to_datetime(
        targets.reshape(-1) - lag_nanoseconds,
        unit="ns",
        utc=True,
    )
    query_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(zones, targets.shape[1]),
            lagged_timestamps,
        ],
        names=["zone_id", "timestamp"],
    )
    values = lookup.reindex(query_index)
    if values.isna().any():
        missing_count = int(values.isna().sum())
        raise ValueError(
            f"missing exact {lag_days}-day lagged observations: {missing_count}"
        )

    return values.to_numpy(dtype=np.float32).reshape(targets.shape)
