"""Train and atomically export the real notebook-03 LightGBM snapshots.

Examples:
    python scripts/export_lightgbm_snapshots.py --origins 1801 --threads 4
    python scripts/export_lightgbm_snapshots.py --origins 1801 1829 1857 1885 --threads 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_engineering import historical_snapshot
from src.features.m5_lightgbm import FEATURE_NAMES, PIPELINE_ID
from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames

SALES_PATH = ROOT / "data/processed/m5_sample_400.csv"
CALENDAR_PATH = ROOT / "data/raw/calendar.csv"
MODEL_DIR = ROOT / "models/lightgbm"
REGISTRY_PATH = ROOT / "models/registry.json"
ORIGINS = (1801, 1829, 1857, 1885)
HORIZON, MIN_HISTORY, ORIGIN_STEP, SEED = 28, 112, 28, 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_joblib(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_panel():
    sales = pd.read_csv(SALES_PATH)
    day_cols = sorted((c for c in sales if c.startswith("d_") and c[2:].isdigit()),
                      key=lambda c: int(c[2:]))
    ids = sales["series_id"].astype(str).tolist()
    if len(ids) != 400 or len(set(ids)) != len(ids):
        raise RuntimeError("Expected the unique frozen 400-series notebook panel")
    values = sales[day_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Training panel contains invalid demand")
    calendar = pd.read_csv(CALENDAR_PATH, usecols=["d", "date"]).drop_duplicates("d").set_index("d")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.reindex(day_cols)["date"]).to_numpy())
    if dates.isna().any():
        raise RuntimeError("Calendar does not cover the notebook panel")
    return ids, values, dates


def origin_frame(values, dates, origin):
    snap = historical_snapshot(values[:, :origin])
    n = len(values)
    rows = {key: np.repeat(value, HORIZON) for key, value in snap.items()}
    rows["series_code"] = np.repeat(np.arange(n, dtype=np.int32), HORIZON)
    rows["horizon"] = np.tile(np.arange(1, HORIZON + 1, dtype=np.int16), n)
    future = dates[origin:origin + HORIZON]
    rows["day_of_week"] = np.tile(future.dayofweek.to_numpy(dtype=np.int8), n)
    rows["month"] = np.tile(future.month.to_numpy(dtype=np.int8), n)
    frame = pd.DataFrame(rows)[FEATURE_NAMES]
    frame["series_code"] = pd.Categorical(frame["series_code"], categories=range(n))
    return frame


def training_data(values, dates, forecast_origin):
    origins = list(range(MIN_HISTORY, forecast_origin - HORIZON + 1, ORIGIN_STEP))
    if not origins or max(origins) + HORIZON > forecast_origin:
        raise RuntimeError("No historically valid training origins")
    frames, targets = [], []
    for origin in origins:
        frames.append(origin_frame(values, dates, origin))
        targets.append(values[:, origin:origin + HORIZON].reshape(-1))
    return pd.concat(frames, ignore_index=True), np.concatenate(targets), origins


def export(origins, threads):
    ids, values, dates = load_panel()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "schema_version": 1, "feature_pipeline_id": PIPELINE_ID,
        "dataset": str(SALES_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": _sha256(SALES_PATH), "training_series_ids": ids,
        "series_id_to_code": {series_id: code for code, series_id in enumerate(ids)},
        "categorical_feature": "series_code", "categories": list(range(len(ids))),
    }
    _atomic_json(mapping, MODEL_DIR / "series_mapping.json")
    params = dict(n_estimators=250, learning_rate=.05, num_leaves=31,
                  min_child_samples=70, colsample_bytree=.9, subsample=.9,
                  subsample_freq=1, reg_lambda=1., random_state=SEED,
                  n_jobs=threads, verbosity=-1)
    snapshots = {}
    for origin in origins:
        started = time.perf_counter()
        X, y, training_origins = training_data(values, dates, origin)
        estimated_mb = (X.memory_usage(deep=True).sum() + y.nbytes) / 1024 ** 2
        print(f"origin={origin} rows={len(X):,} estimated_frame_mb={estimated_mb:.1f}", flush=True)
        result = fit_predict_hurdle_and_direct_frames(
            X, y, origin_frame(values, dates, origin), params=params,
            categorical_features=["series_code"])
        paths = {
            "direct": MODEL_DIR / f"direct_origin_{origin}.joblib",
            "classifier": MODEL_DIR / f"hurdle_classifier_origin_{origin}.joblib",
            "regressor": MODEL_DIR / f"hurdle_regressor_origin_{origin}.joblib",
        }
        _atomic_joblib(result["direct_model"], paths["direct"])
        _atomic_joblib(result["classifier"], paths["classifier"])
        _atomic_joblib(result["quantity_model"], paths["regressor"])
        duration = time.perf_counter() - started
        latest = max(training_origins) + HORIZON
        metadata = {
            "model_type": "DirectLightGBM+HurdleLightGBM", "forecast_origin_day": origin,
            "trained_through_day": latest, "feature_names": FEATURE_NAMES,
            "categorical_encoding": {"series_code": "pandas category; see series_mapping.json"},
            "dataset": mapping["dataset"], "dataset_sha256": mapping["dataset_sha256"],
            "training_series_ids": ids, "training_origins": training_origins,
            "hyperparameters": params, "random_seed": SEED, "training_rows": len(X),
            "positive_training_rows": int((y > 0).sum()), "estimated_frame_mb": estimated_mb,
            "training_duration_seconds": duration, "feature_pipeline_id": PIPELINE_ID,
            "package_versions": {name: version(name) for name in ("lightgbm", "pandas", "numpy", "joblib")},
            "python": platform.python_version(),
        }
        _atomic_json(metadata, MODEL_DIR / f"origin_{origin}.metadata.json")
        snapshots[str(origin)] = {
            "forecast_origin_day": origin, "trained_through_day": latest,
            "feature_pipeline_id": PIPELINE_ID,
            "model_path": f"lightgbm/{paths['direct'].name}",
            "classifier_path": f"lightgbm/{paths['classifier'].name}",
            "regressor_path": f"lightgbm/{paths['regressor'].name}",
            "metadata_path": f"lightgbm/origin_{origin}.metadata.json",
        }
        print(f"origin={origin} complete seconds={duration:.1f}", flush=True)
        del X, y, result
    existing = {}
    if REGISTRY_PATH.exists():
        existing = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for name in ("DirectLightGBM", "HurdleLightGBM"):
        old = existing.get("models", {}).get(name, {}).get("snapshots", {})
        old.update(snapshots)
        existing.setdefault("models", {})[name] = {
            "kind": "direct" if name.startswith("Direct") else "hurdle",
            "feature_names": FEATURE_NAMES, "feature_pipeline_id": PIPELINE_ID,
            "mapping_path": "lightgbm/series_mapping.json", "snapshots": old,
        }
    existing["schema_version"] = 1
    _atomic_json(existing, REGISTRY_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", nargs="+", type=int, choices=ORIGINS, default=list(ORIGINS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    export(args.origins, args.threads)
