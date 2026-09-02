"""Leakage-safe preprocessing for the GEFCom2014 Solar dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_hourly_panel(
    frame: pd.DataFrame,
    accumulated_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Parse, order, and convert accumulated NWP fields to hourly values."""
    required_columns = {"ZONEID", "TIMESTAMP", "POWER", *accumulated_columns}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    panel = frame.copy()
    panel["TIMESTAMP"] = pd.to_datetime(
        panel["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise", utc=True
    )
    panel = panel.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
    if panel.duplicated(["ZONEID", "TIMESTAMP"]).any():
        raise ValueError("duplicate zone-timestamp rows are not allowed")

    time_deltas = panel.groupby("ZONEID", sort=False)["TIMESTAMP"].diff().dropna()
    if not time_deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("each zone must preserve exact hourly continuity")

    panel["forecast_day"] = (
        panel["TIMESTAMP"] - pd.Timedelta(hours=1)
    ).dt.normalize()
    forecast_day_sizes = panel.groupby(["ZONEID", "forecast_day"]).size()
    if not forecast_day_sizes.eq(24).all():
        raise ValueError("each zone forecast day must contain exactly 24 hourly rows")

    group_keys = [panel["ZONEID"], panel["forecast_day"]]
    first_in_forecast_day = panel.groupby(
        ["ZONEID", "forecast_day"], sort=False
    ).cumcount().eq(0)
    audit: dict[str, dict[str, float | int]] = {}

    for column in accumulated_columns:
        raw_increment = panel.groupby(group_keys, sort=False)[column].diff()
        raw_increment.loc[first_in_forecast_day] = panel.loc[
            first_in_forecast_day, column
        ]
        negative = raw_increment < 0
        audit[column] = {
            "negative_count": int(negative.sum()),
            "minimum_raw_increment": float(raw_increment.min()),
        }
        panel[f"{column.lower()}_hourly"] = raw_increment.clip(lower=0)

    panel = panel.rename(columns={column: column.lower() for column in panel.columns})
    panel = panel.rename(columns={"zoneid": "zone_id"})
    return panel, audit


def build_day_ahead_windows(
    panel: pd.DataFrame,
    nwp_columns: Sequence[str],
    lookback_hours: int = 24,
    horizon_hours: int = 24,
    origin_hour: int = 0,
) -> dict[str, np.ndarray]:
    """Build one daily 24-to-24 sample at each valid forecast origin."""
    history_power: list[np.ndarray] = []
    future_nwp: list[np.ndarray] = []
    future_calendar: list[np.ndarray] = []
    target_power: list[np.ndarray] = []
    origins: list[int] = []
    target_timestamps: list[np.ndarray] = []
    zone_ids: list[int] = []

    for zone_id, zone_frame in panel.groupby("zone_id", sort=True):
        zone_frame = zone_frame.sort_values("timestamp").set_index("timestamp")
        candidate_origins = zone_frame.index[zone_frame.index.hour == origin_hour]
        for origin in candidate_origins:
            history_index = pd.date_range(
                origin - pd.Timedelta(hours=lookback_hours - 1),
                origin,
                freq="h",
            )
            future_index = pd.date_range(
                origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="h"
            )
            if not history_index.isin(zone_frame.index).all():
                continue
            if not future_index.isin(zone_frame.index).all():
                continue

            history = zone_frame.loc[history_index]
            future = zone_frame.loc[future_index]
            hour_angle = 2 * np.pi * future_index.hour.to_numpy() / 24
            year_angle = (
                2 * np.pi * (future_index.dayofyear.to_numpy() - 1) / 365.25
            )

            history_power.append(history["power"].to_numpy(np.float32)[:, None])
            future_nwp.append(future[list(nwp_columns)].to_numpy(np.float32))
            future_calendar.append(
                np.column_stack(
                    [
                        np.sin(hour_angle),
                        np.cos(hour_angle),
                        np.sin(year_angle),
                        np.cos(year_angle),
                    ]
                ).astype(np.float32)
            )
            target_power.append(future["power"].to_numpy(np.float32))
            origins.append(origin.value)
            target_timestamps.append(future_index.asi8.copy())
            zone_ids.append(int(zone_id))

    if not origins:
        raise ValueError("no complete day-ahead windows could be constructed")

    return {
        "history_power": np.stack(history_power),
        "future_nwp_raw": np.stack(future_nwp),
        "future_calendar": np.stack(future_calendar),
        "target_power": np.stack(target_power),
        "origin_timestamp_utc": np.asarray(origins, dtype=np.int64),
        "target_timestamp_utc": np.stack(target_timestamps).astype(np.int64),
        "zone_id": np.asarray(zone_ids, dtype=np.int16),
    }


def assign_target_day_splits(
    origin_timestamps_utc: np.ndarray,
    split_dates: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    """Assign samples by the inclusive date of their 24-hour target day."""
    origin_days = pd.to_datetime(
        np.asarray(origin_timestamps_utc, dtype=np.int64), unit="ns", utc=True
    ).normalize()
    masks: dict[str, np.ndarray] = {}
    for split_name, (start_text, end_text) in split_dates.items():
        start = pd.Timestamp(start_text, tz="UTC")
        end = pd.Timestamp(end_text, tz="UTC")
        masks[split_name] = np.asarray(
            (origin_days >= start) & (origin_days <= end), dtype=bool
        )
    return masks


def fit_nwp_scaler(
    train_nwp: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, object]:
    """Fit per-feature standardization statistics on training NWP only."""
    values = np.asarray(train_nwp, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("training NWP must have shape [samples, horizon, features]")
    if values.shape[-1] != len(feature_names):
        raise ValueError("NWP feature count does not match feature names")
    flattened = values.reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return {
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def apply_nwp_scaler(
    values: np.ndarray,
    scaler: Mapping[str, object],
) -> np.ndarray:
    """Apply fixed training statistics without refitting."""
    array = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    if array.shape[-1] != mean.size or mean.shape != scale.shape:
        raise ValueError("NWP values and scaler feature dimensions do not match")
    return ((array - mean) / scale).astype(np.float32)


def save_processed_artifacts(
    output_dir: Path,
    split_windows: Mapping[str, Mapping[str, np.ndarray]],
    scaler: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Persist compressed split arrays and portable JSON metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, arrays in split_windows.items():
        np.savez_compressed(output_dir / f"{split_name}.npz", **arrays)

    json_options = {"ensure_ascii": False, "indent": 2, "sort_keys": True}
    (output_dir / "nwp_scaler.json").write_text(
        json.dumps(dict(scaler), **json_options) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(dict(metadata), **json_options) + "\n", encoding="utf-8"
    )
