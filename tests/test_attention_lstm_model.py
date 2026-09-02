import numpy as np
import torch

from renewable_forecasting.attention_lstm_model import TemporalAttentionLSTM
from renewable_forecasting.lstm_model import build_sequence_arrays


def make_model() -> TemporalAttentionLSTM:
    return TemporalAttentionLSTM(
        future_feature_count=7,
        zone_count=3,
        zone_embedding_dim=4,
        hidden_size=12,
        num_layers=1,
        attention_size=8,
    )


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return (
        torch.randn(5, 24, 1),
        torch.randn(5, 24, 7),
        torch.tensor([0, 1, 2, 0, 1]),
    )


def make_split(target_offset: float = 0.0) -> dict[str, np.ndarray]:
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
        "target_power": (
            np.arange(48, dtype=np.float32).reshape(2, 24) + target_offset
        ),
        "zone_id": np.array([1, 3], dtype=np.int16),
    }


def test_attention_weights_cover_history_and_normalize_per_future_step():
    model = make_model()
    history, future, zones = make_inputs()

    predictions, attention_weights = model.forward_with_attention(
        history,
        future,
        zones,
    )

    assert predictions.shape == (5, 24)
    assert attention_weights.shape == (5, 24, 24)
    assert torch.all(attention_weights >= 0.0)
    torch.testing.assert_close(
        attention_weights.sum(dim=2),
        torch.ones((5, 24)),
    )


def test_default_forward_keeps_the_ordinary_lstm_training_interface():
    model = make_model()
    history, future, zones = make_inputs()

    direct_predictions = model(history, future, zones)
    inspected_predictions, _ = model.forward_with_attention(history, future, zones)

    torch.testing.assert_close(direct_predictions, inspected_predictions)


def test_prediction_loss_reaches_every_attention_lstm_parameter():
    model = make_model()
    history, future, zones = make_inputs()
    targets = torch.linspace(0.0, 1.0, 24).expand(5, -1)

    predictions = model(history, future, zones)
    torch.mean(torch.square(predictions - targets)).backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.any(parameter.grad != 0.0)
    ]
    assert not missing, f"parameters without non-zero prediction gradients: {missing}"


def test_future_targets_never_change_attention_model_inputs_or_predictions():
    original = build_sequence_arrays(make_split(), zone_categories=[1, 2, 3])
    changed = build_sequence_arrays(
        make_split(target_offset=1000.0),
        zone_categories=[1, 2, 3],
    )
    model = make_model()

    original_prediction = model(
        torch.from_numpy(original.history_power),
        torch.from_numpy(original.future_covariates),
        torch.from_numpy(original.zone_indices),
    )
    changed_prediction = model(
        torch.from_numpy(changed.history_power),
        torch.from_numpy(changed.future_covariates),
        torch.from_numpy(changed.zone_indices),
    )

    np.testing.assert_array_equal(original.history_power, changed.history_power)
    np.testing.assert_array_equal(
        original.future_covariates,
        changed.future_covariates,
    )
    torch.testing.assert_close(original_prediction, changed_prediction)


def test_later_future_covariates_do_not_change_earlier_decoder_steps():
    model = make_model()
    history, future, zones = make_inputs()
    changed_future = future.clone()
    changed_future[:, 12:, :] += 100.0

    original_predictions, original_attention = model.forward_with_attention(
        history,
        future,
        zones,
    )
    changed_predictions, changed_attention = model.forward_with_attention(
        history,
        changed_future,
        zones,
    )

    torch.testing.assert_close(
        original_predictions[:, :12],
        changed_predictions[:, :12],
    )
    torch.testing.assert_close(
        original_attention[:, :12],
        changed_attention[:, :12],
    )
