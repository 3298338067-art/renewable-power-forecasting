import numpy as np
import pytest
import torch

from renewable_forecasting.lstm_model import (
    CovariateOnlyLSTM,
    build_sequence_arrays,
)


def _make_split() -> dict[str, np.ndarray]:
    samples = 3
    horizon = 24
    return {
        "history_power": np.arange(
            samples * horizon, dtype=np.float32
        ).reshape(samples, horizon, 1),
        "future_nwp_scaled": np.full(
            (samples, horizon, 12), 7.0, dtype=np.float32
        ),
        "future_calendar": np.full(
            (samples, horizon, 4), 3.0, dtype=np.float32
        ),
        "target_power": np.zeros((samples, horizon), dtype=np.float32),
        "zone_id": np.array([1, 2, 3], dtype=np.int16),
    }


def test_history_only_excludes_every_nwp_feature():
    original = _make_split()
    changed = _make_split()
    changed["future_nwp_scaled"][:] = -19.0

    original_arrays = build_sequence_arrays(
        original,
        [1, 2, 3],
        input_variant="history_only",
    )
    changed_arrays = build_sequence_arrays(
        changed,
        [1, 2, 3],
        input_variant="history_only",
    )

    assert original_arrays.future_covariates.shape == (3, 24, 4)
    np.testing.assert_array_equal(
        original_arrays.future_covariates,
        original["future_calendar"],
    )
    np.testing.assert_array_equal(
        original_arrays.future_covariates,
        changed_arrays.future_covariates,
    )


def test_nwp_only_arrays_keep_nwp_calendar_and_not_target():
    original = _make_split()
    changed = _make_split()
    changed["target_power"][:] = 1.0

    original_arrays = build_sequence_arrays(
        original,
        [1, 2, 3],
        input_variant="nwp_only",
    )
    changed_arrays = build_sequence_arrays(
        changed,
        [1, 2, 3],
        input_variant="nwp_only",
    )

    assert original_arrays.future_covariates.shape == (3, 24, 16)
    np.testing.assert_array_equal(
        original_arrays.future_covariates,
        changed_arrays.future_covariates,
    )


def test_nwp_only_model_has_no_history_information_path():
    torch.manual_seed(5)
    model = CovariateOnlyLSTM(
        future_feature_count=16,
        zone_count=3,
        zone_embedding_dim=4,
        hidden_size=8,
        num_layers=1,
    )
    model.eval()
    history_a = torch.zeros(3, 24, 1)
    history_b = torch.full((3, 24, 1), 999.0)
    future = torch.randn(3, 24, 16)
    zones = torch.tensor([0, 1, 2])

    prediction_a = model(history_a, future, zones)
    prediction_b = model(history_b, future, zones)

    assert not hasattr(model, "encoder")
    assert all("encoder" not in name for name, _ in model.named_parameters())
    torch.testing.assert_close(prediction_a, prediction_b, rtol=0.0, atol=0.0)
    assert prediction_a.shape == (3, 24)


def test_input_variant_rejects_unknown_name():
    with pytest.raises(ValueError, match="input variant"):
        build_sequence_arrays(
            _make_split(),
            [1, 2, 3],
            input_variant="future_target_only",
        )
