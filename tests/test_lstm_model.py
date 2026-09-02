import numpy as np
import pytest
import torch

from renewable_forecasting.lstm_model import (
    Seq2SeqLSTM,
    build_sequence_arrays,
)


def make_split() -> dict[str, np.ndarray]:
    return {
        "history_power": np.arange(48, dtype=np.float32).reshape(2, 24, 1),
        "future_nwp_scaled": np.arange(
            2 * 24 * 3,
            dtype=np.float32,
        ).reshape(2, 24, 3),
        "future_calendar": np.arange(
            2 * 24 * 4,
            dtype=np.float32,
        ).reshape(2, 24, 4),
        "target_power": np.arange(48, dtype=np.float32).reshape(2, 24),
        "zone_id": np.array([1, 3], dtype=np.int16),
    }


def test_build_sequence_arrays_uses_scaled_nwp_and_maps_zones():
    arrays = build_sequence_arrays(make_split(), zone_categories=[1, 2, 3])

    assert arrays.history_power.shape == (2, 24, 1)
    assert arrays.future_covariates.shape == (2, 24, 7)
    assert arrays.targets.shape == (2, 24)
    np.testing.assert_array_equal(arrays.zone_indices, [0, 2])
    np.testing.assert_array_equal(
        arrays.future_covariates[:, :, :3],
        make_split()["future_nwp_scaled"],
    )


def test_build_sequence_arrays_rejects_unknown_zone():
    split = make_split()
    split["zone_id"][0] = 4

    with pytest.raises(ValueError, match="unknown zone"):
        build_sequence_arrays(split, zone_categories=[1, 2, 3])


def test_future_targets_never_change_model_inputs():
    original = make_split()
    changed = make_split()
    changed["target_power"] += 1000

    original_arrays = build_sequence_arrays(original, zone_categories=[1, 2, 3])
    changed_arrays = build_sequence_arrays(changed, zone_categories=[1, 2, 3])

    np.testing.assert_array_equal(
        original_arrays.history_power,
        changed_arrays.history_power,
    )
    np.testing.assert_array_equal(
        original_arrays.future_covariates,
        changed_arrays.future_covariates,
    )
    assert not np.array_equal(original_arrays.targets, changed_arrays.targets)


def test_seq2seq_lstm_outputs_one_value_per_future_hour():
    model = Seq2SeqLSTM(
        future_feature_count=7,
        zone_count=3,
        zone_embedding_dim=4,
        hidden_size=16,
        num_layers=1,
    )
    history = torch.randn(5, 24, 1)
    future = torch.randn(5, 24, 7)
    zones = torch.tensor([0, 1, 2, 0, 1])

    predictions = model(history, future, zones)

    assert predictions.shape == (5, 24)
    predictions.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
