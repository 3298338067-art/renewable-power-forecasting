"""Leakage-safe tensors and ordinary Seq2Seq LSTM for solar forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SequenceModelArrays:
    """Validated arrays consumed by sequence forecasting models."""

    history_power: np.ndarray
    future_covariates: np.ndarray
    targets: np.ndarray
    zone_indices: np.ndarray


def build_sequence_arrays(
    split: Mapping[str, np.ndarray],
    zone_categories: Sequence[int],
    *,
    input_variant: str = "full",
) -> SequenceModelArrays:
    """Build model inputs from past power and forecast-origin-known covariates."""
    valid_variants = {"full", "history_only", "nwp_only"}
    if input_variant not in valid_variants:
        raise ValueError(
            f"unknown input variant {input_variant!r}; "
            f"expected one of {sorted(valid_variants)}"
        )
    required = {
        "history_power",
        "future_nwp_scaled",
        "future_calendar",
        "target_power",
        "zone_id",
    }
    missing = sorted(required.difference(split))
    if missing:
        raise ValueError(f"processed split is missing arrays: {missing}")

    history = np.asarray(split["history_power"], dtype=np.float32)
    future_nwp = np.asarray(split["future_nwp_scaled"], dtype=np.float32)
    calendar = np.asarray(split["future_calendar"], dtype=np.float32)
    targets = np.asarray(split["target_power"], dtype=np.float32)
    zones = np.asarray(split["zone_id"])

    if history.ndim != 3 or history.shape[2] != 1:
        raise ValueError("history_power must have shape [samples, history, 1]")
    sample_count = history.shape[0]
    if future_nwp.ndim != 3 or future_nwp.shape[0] != sample_count:
        raise ValueError(
            "future_nwp_scaled must have shape [samples, horizon, features]"
        )
    horizon_hours = future_nwp.shape[1]
    if (
        calendar.ndim != 3
        or calendar.shape[:2] != (sample_count, horizon_hours)
    ):
        raise ValueError(
            "future_calendar must match samples and forecast horizon"
        )
    if targets.shape != (sample_count, horizon_hours):
        raise ValueError("target_power must match samples and forecast horizon")
    if zones.ndim != 1 or zones.size != sample_count:
        raise ValueError("zone_id must contain one value per sample")

    categories = tuple(int(zone) for zone in zone_categories)
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("zone categories must be unique and non-empty")
    zone_to_index = {zone: index for index, zone in enumerate(categories)}
    unknown = sorted(set(int(zone) for zone in np.unique(zones)) - set(categories))
    if unknown:
        raise ValueError(f"unknown zone IDs in processed split: {unknown}")
    zone_indices = np.asarray(
        [zone_to_index[int(zone)] for zone in zones],
        dtype=np.int64,
    )

    arrays = (history, future_nwp, calendar, targets)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("sequence model arrays contain non-finite values")
    if input_variant == "history_only":
        future_covariates = calendar
    else:
        future_covariates = np.concatenate(
            [future_nwp, calendar],
            axis=2,
        )
    future_covariates = future_covariates.astype(np.float32, copy=False)
    return SequenceModelArrays(
        history_power=np.ascontiguousarray(history),
        future_covariates=np.ascontiguousarray(future_covariates),
        targets=np.ascontiguousarray(targets),
        zone_indices=np.ascontiguousarray(zone_indices),
    )


class Seq2SeqLSTM(nn.Module):
    """Encode historical power and decode known future covariates."""

    def __init__(
        self,
        future_feature_count: int,
        zone_count: int,
        zone_embedding_dim: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if min(
            future_feature_count,
            zone_count,
            zone_embedding_dim,
            hidden_size,
            num_layers,
        ) <= 0:
            raise ValueError("all LSTM dimensions must be positive")
        self.zone_count = zone_count
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.zone_embedding = nn.Embedding(zone_count, zone_embedding_dim)
        self.decoder = nn.LSTM(
            input_size=future_feature_count + zone_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        history_power: torch.Tensor,
        future_covariates: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> torch.Tensor:
        if history_power.ndim != 3 or history_power.shape[2] != 1:
            raise ValueError("history_power must have shape [batch, history, 1]")
        if future_covariates.ndim != 3:
            raise ValueError(
                "future_covariates must have shape [batch, horizon, features]"
            )
        if (
            future_covariates.shape[0] != history_power.shape[0]
            or zone_indices.shape != (history_power.shape[0],)
        ):
            raise ValueError("all LSTM inputs must have the same batch size")
        if torch.any(zone_indices < 0) or torch.any(
            zone_indices >= self.zone_count
        ):
            raise ValueError("zone index is outside the embedding range")

        _, encoder_state = self.encoder(history_power)
        zone_features = self.zone_embedding(zone_indices).unsqueeze(1)
        zone_features = zone_features.expand(
            -1,
            future_covariates.shape[1],
            -1,
        )
        decoder_inputs = torch.cat([future_covariates, zone_features], dim=2)
        decoder_outputs, _ = self.decoder(decoder_inputs, encoder_state)
        return self.output_head(decoder_outputs).squeeze(-1)


class CovariateOnlyLSTM(nn.Module):
    """Decode known future covariates without any historical-power path."""

    def __init__(
        self,
        future_feature_count: int,
        zone_count: int,
        zone_embedding_dim: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if min(
            future_feature_count,
            zone_count,
            zone_embedding_dim,
            hidden_size,
            num_layers,
        ) <= 0:
            raise ValueError("all LSTM dimensions must be positive")
        self.zone_count = zone_count
        self.zone_embedding = nn.Embedding(zone_count, zone_embedding_dim)
        self.decoder = nn.LSTM(
            input_size=future_feature_count + zone_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        history_power: torch.Tensor,
        future_covariates: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> torch.Tensor:
        if history_power.ndim != 3 or history_power.shape[2] != 1:
            raise ValueError("history_power must have shape [batch, history, 1]")
        if future_covariates.ndim != 3:
            raise ValueError(
                "future_covariates must have shape [batch, horizon, features]"
            )
        if (
            future_covariates.shape[0] != history_power.shape[0]
            or zone_indices.shape != (history_power.shape[0],)
        ):
            raise ValueError("all LSTM inputs must have the same batch size")
        if torch.any(zone_indices < 0) or torch.any(
            zone_indices >= self.zone_count
        ):
            raise ValueError("zone index is outside the embedding range")

        zone_features = self.zone_embedding(zone_indices).unsqueeze(1)
        zone_features = zone_features.expand(
            -1,
            future_covariates.shape[1],
            -1,
        )
        decoder_inputs = torch.cat([future_covariates, zone_features], dim=2)
        decoder_outputs, _ = self.decoder(decoder_inputs)
        return self.output_head(decoder_outputs).squeeze(-1)
