import json
from pathlib import Path

import pytest

import scripts.evaluate_baselines as evaluate_baselines


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "day_ahead.yaml"
PROCESSED_DIR = ROOT / "data" / "processed" / "day_ahead"
INTERIM_PATH = ROOT / "data" / "interim" / "gefcom2014_solar_hourly.csv.gz"


@pytest.mark.skipif(
    not (PROCESSED_DIR / "test.npz").is_file() or not INTERIM_PATH.is_file(),
    reason="official processed artifacts are unavailable",
)
def test_official_baseline_evaluation_saves_metrics_and_figures(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"

    results = evaluate_baselines.run_evaluation(
        CONFIG_PATH,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
    )

    assert results["evaluated_splits"] == ["validation", "test"]
    assert results["time_standard"] == "UTC"
    assert results["irradiance_active_definition"] == (
        "forecast VAR169 hourly increment > 0 J m^-2"
    )
    assert set(results["validation"]) == {
        "daily_persistence",
        "seasonal_persistence_7d",
    }
    assert results["validation"]["daily_persistence"]["overall"]["count"] == 6480
    assert results["test"]["daily_persistence"]["overall"]["count"] == 6552

    for split_name in ("validation", "test"):
        for baseline_name in (
            "daily_persistence",
            "seasonal_persistence_7d",
        ):
            summary = results[split_name][baseline_name]
            assert summary["overall"]["mae"] >= 0
            assert summary["overall"]["rmse"] >= 0
            assert summary["overall"]["nrmse_capacity_percent"] >= 0
            assert summary["irradiance_active"]["count"] > 0
            assert summary["irradiance_active"]["count"] < summary["overall"]["count"]
            assert summary["irradiance_active"]["nrmse_capacity_percent"] >= 0
            assert set(summary["by_zone"]) == {"1", "2", "3"}
            assert len(summary["by_horizon"]) == 24

    saved_results = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_results == results
    for figure_name in (
        "baseline_forecast_example.png",
        "baseline_horizon_rmse.png",
    ):
        figure_path = figure_dir / figure_name
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0
