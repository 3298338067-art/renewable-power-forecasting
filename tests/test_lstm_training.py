import numpy as np
import pytest
import torch
from torch import nn

from renewable_forecasting.lstm_model import SequenceModelArrays
from renewable_forecasting.lstm_training import (
    TrainingHistory,
    fit_lstm,
    predict_lstm,
    set_reproducible_seed,
    validation_has_improved,
)


def make_arrays() -> SequenceModelArrays:
    return SequenceModelArrays(
        history_power=np.zeros((3, 24, 1), dtype=np.float32),
        future_covariates=np.zeros((3, 24, 5), dtype=np.float32),
        targets=np.zeros((3, 24), dtype=np.float32),
        zone_indices=np.array([0, 1, 2], dtype=np.int64),
    )


class ConstantModel(nn.Module):
    def forward(self, history, future, zones):
        del history, zones
        return torch.full(
            future.shape[:2],
            1.25,
            dtype=future.dtype,
            device=future.device,
        )


class BiasModel(nn.Module):
    def __init__(self, initial_bias: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

    def forward(self, history, future, zones):
        del history, zones
        return self.bias.expand(future.shape[:2])


def test_set_reproducible_seed_resets_numpy_and_torch():
    set_reproducible_seed(42, deterministic_algorithms=True)
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    set_reproducible_seed(42, deterministic_algorithms=True)

    np.testing.assert_array_equal(np.random.rand(3), first_numpy)
    torch.testing.assert_close(torch.rand(3), first_torch)


def test_validation_improvement_respects_minimum_delta():
    assert validation_has_improved(0.89, best_loss=1.0, min_delta=0.1)
    assert not validation_has_improved(0.91, best_loss=1.0, min_delta=0.1)


def test_predict_lstm_batches_and_applies_fixed_bounds():
    predictions = predict_lstm(
        ConstantModel(),
        make_arrays(),
        batch_size=2,
        device=torch.device("cpu"),
        clip_bounds=(0.0, 1.0),
        num_workers=0,
    )

    assert predictions.shape == (3, 24)
    np.testing.assert_array_equal(predictions, np.ones((3, 24)))


def test_fit_lstm_records_losses_and_reduces_training_error():
    train_arrays = make_arrays()
    train_arrays.targets.fill(1.0)
    model = BiasModel()

    history = fit_lstm(
        model,
        train_arrays,
        train_arrays,
        batch_size=3,
        learning_rate=0.1,
        weight_decay=0.0,
        max_epochs=4,
        patience=4,
        min_delta=0.0,
        gradient_clip_norm=1.0,
        device=torch.device("cpu"),
        seed=42,
        num_workers=0,
    )

    assert isinstance(history, TrainingHistory)
    assert history.epochs_completed == 4
    assert len(history.train_losses) == history.epochs_completed
    assert len(history.validation_losses) == history.epochs_completed
    assert history.train_losses[-1] < history.train_losses[0]
    assert history.best_epoch == int(np.argmin(history.validation_losses)) + 1


def test_fit_lstm_early_stops_and_restores_best_validation_weights():
    train_arrays = make_arrays()
    train_arrays.targets.fill(1.0)
    validation_arrays = make_arrays()
    validation_arrays.targets.fill(0.0)
    model = BiasModel(initial_bias=0.1)

    history = fit_lstm(
        model,
        train_arrays,
        validation_arrays,
        batch_size=3,
        learning_rate=0.1,
        weight_decay=0.0,
        max_epochs=20,
        patience=2,
        min_delta=0.0,
        gradient_clip_norm=1.0,
        device=torch.device("cpu"),
        seed=42,
        num_workers=0,
    )

    assert history.stopped_early
    assert history.epochs_completed == 3
    restored_predictions = predict_lstm(
        model,
        validation_arrays,
        batch_size=3,
        device=torch.device("cpu"),
        clip_bounds=(-10.0, 10.0),
        num_workers=0,
    )
    restored_loss = float(np.mean(np.square(restored_predictions)))
    assert restored_loss == pytest.approx(min(history.validation_losses))
