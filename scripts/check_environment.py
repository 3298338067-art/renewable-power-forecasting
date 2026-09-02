from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "xgboost": "xgboost",
    "torch": "torch",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        print("ERROR: use Python 3.10, 3.11, or 3.12 for this project.")
        return 1

    missing: list[str] = []
    versions: dict[str, str] = {}
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            versions[package_name] = getattr(module, "__version__", "installed")
        except ImportError:
            missing.append(package_name)

    if missing:
        print("ERROR: missing packages: " + ", ".join(missing))
        return 1

    print("Packages:")
    for package_name, version in sorted(versions.items()):
        print(f"  {package_name}: {version}")

    torch = importlib.import_module("torch")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA build: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU is unavailable; CPU execution remains supported.")

    config_path = ROOT / "configs" / "day_ahead.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    task = config["task"]
    print(
        "Task: "
        f"{task['lookback_hours']}h history -> "
        f"{task['forecast_horizon_hours']}h forecast"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
