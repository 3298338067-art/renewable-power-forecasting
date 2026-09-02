"""Reproducible training and inference utilities for sequence models."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from renewable_forecasting.lstm_model import SequenceModelArrays


@dataclass(frozen=True)
class TrainingHistory:
    """Loss trajectory and validation-selected epoch for one training run."""

    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    stopped_early: bool


def set_reproducible_seed(
    seed: int,
    deterministic_algorithms: bool,
) -> None:
    """Reset Python, NumPy, and PyTorch random state for a model run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.backends.cudnn.benchmark = not deterministic_algorithms


def validation_has_improved(
    current_loss: float,
    best_loss: float,
    min_delta: float,
) -> bool:
    """Return whether validation loss beat the previous best by min_delta."""
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    return current_loss < best_loss - min_delta


def _as_dataset(arrays: SequenceModelArrays) -> TensorDataset:
    sample_count = arrays.history_power.shape[0]
    if sample_count == 0:
        raise ValueError("sequence arrays must contain at least one sample")
    if (
        arrays.future_covariates.shape[0] != sample_count
        or arrays.targets.shape[0] != sample_count
        or arrays.zone_indices.shape != (sample_count,)
    ):
        raise ValueError("all sequence arrays must have the same sample count")
    return TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(arrays.history_power, dtype=np.float32)
        ),
        torch.from_numpy(
            np.ascontiguousarray(arrays.future_covariates, dtype=np.float32)
        ),
        torch.from_numpy(np.ascontiguousarray(arrays.targets, dtype=np.float32)),
        torch.from_numpy(
            np.ascontiguousarray(arrays.zone_indices, dtype=np.int64)
        ),
    )


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for history, future, targets, zones in loader:
            history = history.to(device)
            future = future.to(device)
            targets = targets.to(device)
            zones = zones.to(device)
            batch_loss = loss_function(model(history, future, zones), targets)
            batch_size = history.shape[0]
            total_loss += float(batch_loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def predict_lstm(
    model: nn.Module,
    arrays: SequenceModelArrays,
    *,
    batch_size: int,
    device: torch.device | str,
    clip_bounds: tuple[float, float],
    num_workers: int,
) -> np.ndarray:
    """Predict sequence arrays in batches and clip normalized power."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lower, upper = (float(clip_bounds[0]), float(clip_bounds[1]))
    if lower >= upper:
        raise ValueError("clip bounds must be strictly increasing")

    resolved_device = torch.device(device)
    loader = DataLoader(
        _as_dataset(arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    was_training = model.training
    model.to(resolved_device)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for history, future, _, zones in loader:
            prediction = model(
                history.to(resolved_device),
                future.to(resolved_device),
                zones.to(resolved_device),
            )
            batches.append(prediction.detach().cpu().numpy())
    model.train(was_training)
    return np.clip(np.concatenate(batches, axis=0), lower, upper)


def fit_lstm(
    model: nn.Module,
    train_arrays: SequenceModelArrays,
    validation_arrays: SequenceModelArrays,
    *,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    gradient_clip_norm: float,
    device: torch.device | str,
    seed: int,
    num_workers: int,
) -> TrainingHistory:
    """Fit using train data, select only on validation loss, and restore best."""
    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")

    set_reproducible_seed(seed, deterministic_algorithms=True)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    training_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _as_dataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        generator=training_generator,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        _as_dataset(validation_arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.MSELoss()
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        for history, future, targets, zones in train_loader:
            history = history.to(resolved_device)
            future = future.to(resolved_device)
            targets = targets.to(resolved_device)
            zones = zones.to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss_function(model(history, future, zones), targets)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            current_batch_size = history.shape[0]
            total_train_loss += float(batch_loss.item()) * current_batch_size
            total_train_samples += current_batch_size

        train_loss = total_train_loss / total_train_samples
        validation_loss = _mean_loss(
            model,
            validation_loader,
            loss_function,
            resolved_device,
        )
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise FloatingPointError("LSTM training produced a non-finite loss")
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_has_improved(
            validation_loss,
            best_validation_loss,
            min_delta,
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training finished without a finite validation checkpoint")
    model.load_state_dict(best_state)
    epochs_completed = len(train_losses)
    return TrainingHistory(
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=epochs_completed,
        stopped_early=epochs_completed < max_epochs,
    )
