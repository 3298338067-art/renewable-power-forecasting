from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "configs/day_ahead.yaml",
        "data/README.md",
        "docs/technical_report_en.md",
        "docs/technical_report_zh.md",
        "scripts/check_environment.py",
        "scripts/train_xgboost.py",
    ]

    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"Missing required project files: {missing}"


def test_day_ahead_task_is_fixed_and_leakage_safe():
    config_path = ROOT / "configs/day_ahead.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    task = config["task"]
    data = config["data"]
    evaluation = config["evaluation"]
    xgboost = config["xgboost"]
    lstm = config["lstm"]
    attention_lstm = config["attention_lstm"]
    lstm_ablations = config["lstm_ablations"]
    exploratory_analysis = config["exploratory_analysis"]

    assert task["lookback_hours"] == 24
    assert task["forecast_horizon_hours"] == 24
    assert task["weather_input"] == "forecast_origin_available_nwp"
    assert data["split_method"] == "chronological"
    assert data["scaler_fit_scope"] == "train_only"
    assert data["raw_csv"].endswith("Task 15/predictors15.csv")
    assert data["forecast_origin_hour"] == 0
    assert data["split_dates"]["train"] == ["2012-04-02", "2013-12-31"]
    assert data["split_dates"]["validation"] == ["2014-01-01", "2014-03-31"]
    assert data["split_dates"]["test"] == ["2014-04-01", "2014-06-30"]
    assert data["accumulated_nwp"] == ["VAR169", "VAR175", "VAR178", "VAR228"]
    assert evaluation["random_seeds"] == [42, 2026, 3407]
    assert "nrmse_capacity_percent" in evaluation["metrics"]
    assert "irradiance_active_rmse" in evaluation["metrics"]
    assert xgboost["device"] == "cuda"
    assert xgboost["selection_metric"] == "validation_rmse"
    assert xgboost["prediction_clip"] == [0.0, 1.0]
    assert xgboost["selection_seed"] == 42
    assert len(xgboost["candidates"]) >= 2
    assert lstm["hidden_size"] == 64
    assert lstm["batch_size"] == 128
    assert lstm["max_epochs"] == 100
    assert lstm["early_stopping_patience"] == 12
    assert lstm["prediction_clip"] == [0.0, 1.0]
    assert lstm["device"] == "cuda"
    fair_keys = {
        "device",
        "hidden_size",
        "num_layers",
        "zone_embedding_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "gradient_clip_norm",
        "loss",
        "prediction_clip",
        "deterministic_algorithms",
        "num_workers",
    }
    assert {key: attention_lstm[key] for key in fair_keys} == {
        key: lstm[key] for key in fair_keys
    }
    assert attention_lstm["attention_size"] == lstm["hidden_size"]
    assert lstm_ablations["post_hoc_exploratory"] is True
    assert lstm_ablations["variants"] == ["history_only", "nwp_only"]
    assert (
        lstm_ablations["full_input_control_metrics"]
        == "artifacts/lstm/metrics.json"
    )
    assert not fair_keys.intersection(lstm_ablations)
    assert exploratory_analysis["post_hoc_exploratory"] is True
    assert exploratory_analysis["high_power_quantile"] == 0.75
    assert exploratory_analysis["ramp_quantile"] == 0.90
    assert exploratory_analysis["bootstrap_block_lengths_days"] == [3, 7, 14]
    assert exploratory_analysis["primary_block_length_days"] == 7
    assert exploratory_analysis["bootstrap_resamples"] == 2000
    assert exploratory_analysis["bootstrap_seed"] == 20260901
