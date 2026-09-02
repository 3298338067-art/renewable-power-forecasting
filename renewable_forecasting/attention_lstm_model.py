"""Temporal-attention Seq2Seq LSTM for leakage-safe solar forecasting."""

from __future__ import annotations

import torch
from torch import nn


class AdditiveTemporalAttention(nn.Module):
    """Bahdanau attention over historical encoder states."""

    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or attention_size <= 0:
            raise ValueError("attention dimensions must be positive")
        self.key_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.score_projection = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError("attention query must have shape [batch, hidden]")
        if encoder_outputs.ndim != 3:
            raise ValueError(
                "encoder outputs must have shape [batch, history, hidden]"
            )
        if (
            query.shape[0] != encoder_outputs.shape[0]
            or query.shape[1] != encoder_outputs.shape[2]
        ):
            raise ValueError("attention query and encoder outputs are incompatible")

        energy = torch.tanh(
            self.key_projection(encoder_outputs)
            + self.query_projection(query).unsqueeze(1)
        )
        scores = self.score_projection(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class TemporalAttentionLSTM(nn.Module):
    """Attend to 24 historical states before each future decoder step."""

    def __init__(
        self,
        future_feature_count: int,
        zone_count: int,
        zone_embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        attention_size: int,
    ) -> None:
        super().__init__()
        if min(
            future_feature_count,
            zone_count,
            zone_embedding_dim,
            hidden_size,
            num_layers,
            attention_size,
        ) <= 0:
            raise ValueError("all Attention-LSTM dimensions must be positive")
        self.zone_count = zone_count
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = AdditiveTemporalAttention(hidden_size, attention_size)
        self.zone_embedding = nn.Embedding(zone_count, zone_embedding_dim)
        self.decoder = nn.LSTM(
            input_size=(
                future_feature_count + zone_embedding_dim + hidden_size
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(2 * hidden_size, 1)

    def _validate_inputs(
        self,
        history_power: torch.Tensor,
        future_covariates: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> None:
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
            raise ValueError("all Attention-LSTM inputs must share a batch size")
        if torch.any(zone_indices < 0) or torch.any(
            zone_indices >= self.zone_count
        ):
            raise ValueError("zone index is outside the embedding range")

    def forward_with_attention(
        self,
        history_power: torch.Tensor,
        future_covariates: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return forecasts and weights [batch, future, history]."""
        self._validate_inputs(history_power, future_covariates, zone_indices)
        encoder_outputs, decoder_state = self.encoder(history_power)
        zone_features = self.zone_embedding(zone_indices)
        predictions: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []

        for future_index in range(future_covariates.shape[1]):
            query = decoder_state[0][-1]
            context, weights = self.attention(query, encoder_outputs)
            decoder_input = torch.cat(
                [
                    future_covariates[:, future_index, :],
                    zone_features,
                    context,
                ],
                dim=1,
            ).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(
                decoder_input,
                decoder_state,
            )
            prediction = self.output_head(
                torch.cat([decoder_output[:, 0, :], context], dim=1)
            ).squeeze(-1)
            predictions.append(prediction)
            attention_weights.append(weights)

        return (
            torch.stack(predictions, dim=1),
            torch.stack(attention_weights, dim=1),
        )

    def forward(
        self,
        history_power: torch.Tensor,
        future_covariates: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> torch.Tensor:
        predictions, _ = self.forward_with_attention(
            history_power,
            future_covariates,
            zone_indices,
        )
        return predictions
