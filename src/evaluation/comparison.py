"""Matched-observation forecast comparison."""
from __future__ import annotations
import pandas as pd

KEYS = ["scenario", "series_id", "forecast_origin", "target_day", "horizon_step"]
REQUIRED = KEYS + ["model", "actual", "prediction"]


def validate_prediction_records(frame):
    missing = set(REQUIRED)-set(frame.columns)
    if missing: raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if frame[REQUIRED].isna().any().any(): raise ValueError("prediction records contain missing values")
    if frame.duplicated(KEYS+["model"]).any(): raise ValueError("duplicate model predictions")
    return frame


def assert_matched_predictions(frame):
    frame = validate_prediction_records(frame)
    reference = None
    for model, group in frame.groupby("model", sort=True):
        current = group[KEYS+["actual"]].sort_values(KEYS).reset_index(drop=True)
        if reference is None: reference = current
        elif not current.equals(reference): raise ValueError(f"model {model} is not evaluated on identical observations")
    return True
