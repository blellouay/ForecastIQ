"""Saved-ML inference and as-of historical comparisons, without training.

IMPORTANT: Registry and joblib artifacts must come from a trusted local source.
FeatureProvider is user-supplied project code, accepting *history as of origin* only.
No global test-set leaderboard or future target can be used for selection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.agent_tools.forecasting_tools import ToolError, _history, _horizon
from .model_router import validation_cutoffs, select_by_rolling_validation

ML_NAMES = ("DirectLightGBM", "HurdleLightGBM")
FeatureProvider = Callable[[str, list[float], int, int, list[str]], Any]


def _registry(path: str | Path) -> tuple[dict, Path]:
    p = Path(path).resolve()
    if not p.is_file():
        raise ToolError(f"Saved-ML registry does not exist: {p}")
    try:
        document = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError("Saved-ML registry is not valid JSON") from exc
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), dict):
        raise ToolError("Registry requires schema_version=1 and a models object")
    return document, p.parent


def _artifact(base: Path, value: str):
    if not isinstance(value, str) or not value:
        raise ToolError("Artifact path must be nonempty")
    p = Path(value)
    p = p if p.is_absolute() else base / p
    p = p.resolve()
    if not p.is_file():
        raise ToolError(f"Saved ML artifact missing: {p}")
    return p


def _snapshot(document: dict, model: str, origin: int) -> dict:
    entry = document["models"].get(model)
    if not isinstance(entry, dict) or entry.get("kind") not in ("direct", "hurdle"):
        raise ToolError(f"Model {model} is not registered with a supported kind")
    if (model == "DirectLightGBM") != (entry["kind"] == "direct"):
        raise ToolError("Model name and kind disagree")
    features = entry.get("feature_names")
    if not isinstance(features, list) or not features or len(set(features)) != len(features) or not all(isinstance(x, str) and x for x in features):
        raise ToolError("Missing or invalid ordered feature_names")
    snapshot = entry.get("snapshots", {}).get(str(origin))
    if not isinstance(snapshot, dict):
        raise ToolError(f"No as-of snapshot for {model} at origin {origin}; no training will be started")
    cutoff = snapshot.get("trained_through_day")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff > origin or cutoff < 1:
        raise ToolError(f"Invalid training cutoff for {model} at origin {origin}")
    if snapshot.get("forecast_origin_day") != origin:
        raise ToolError("Snapshot origin mismatch")
    if snapshot.get("feature_pipeline_id") != entry.get("feature_pipeline_id") or not entry.get("feature_pipeline_id"):
        raise ToolError("Feature pipeline identifier absent or mismatched")
    return {**snapshot, "kind": entry["kind"], "features": features}


def predict_snapshot(
    series_id: str, history: list[float], origin: int, horizon: int, model: str,
    registry_path: str | Path, feature_provider: FeatureProvider,
) -> dict[str, Any]:
    """Predict at one origin. Exact-origin snapshots make stale/future models fail closed."""
    h = _horizon(horizon)
    historical = _history(history)
    if not isinstance(origin, int) or isinstance(origin, bool) or origin < 1:
        raise ToolError("forecast origin must be a positive absolute M5 day")
    if not callable(feature_provider):
        raise ToolError("A trusted as-of feature provider is required for saved ML")
    doc, base = _registry(registry_path)
    snapshot = _snapshot(doc, model, origin)
    # Provider receives only truncated history, never raw future sales or the full dataset.
    frame = feature_provider(series_id, historical.tolist(), origin, h, list(snapshot["features"]))
    try:
        import pandas as pd
        if not isinstance(frame, pd.DataFrame) or len(frame) != h:
            raise ToolError(f"Feature provider must return a DataFrame with exactly {h} ordered rows")
        if list(frame.columns) != snapshot["features"]:
            raise ToolError("ML feature columns/order differ from the training schema")
        if frame.isnull().any().any():
            raise ToolError("ML feature provider returned null values")
        import joblib
        if snapshot["kind"] == "direct":
            model_file = _artifact(base, snapshot.get("model_path"))
            predictor = joblib.load(model_file)  # TRUSTED LOCAL FILES ONLY
            raw = predictor.predict(frame)
            files = [str(model_file)]
        else:
            classifier_file = _artifact(base, snapshot.get("classifier_path"))
            regressor_file = _artifact(base, snapshot.get("regressor_path"))
            classifier, regressor = joblib.load(classifier_file), joblib.load(regressor_file)
            if not hasattr(classifier, "predict_proba"):
                raise ToolError("Hurdle classifier must implement predict_proba")
            probabilities = np.asarray(classifier.predict_proba(frame), dtype=float)
            if probabilities.shape != (h, 2):
                raise ToolError("Hurdle classifier probability output must be (horizon, 2)")
            # Positive label is explicit: classes_ order is not assumed.
            classes = getattr(classifier, "classes_", None)
            if classes is None or 1 not in list(classes):
                raise ToolError("Hurdle classifier needs positive class label 1")
            p = probabilities[:, list(classes).index(1)]
            quantity = np.asarray(regressor.predict(frame), dtype=float)
            if quantity.shape != (h,) or not np.isfinite(quantity).all() or (quantity < 0).any():
                raise ToolError("Hurdle positive-quantity forecast is invalid")
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ToolError("Hurdle occurrence probability is invalid")
            raw = p * quantity
            files = [str(classifier_file), str(regressor_file)]
        forecast = np.asarray(raw, dtype=float)
        if forecast.shape != (h,) or not np.isfinite(forecast).all() or (forecast < 0).any():
            raise ToolError("Saved ML prediction must be finite and nonnegative; no silent clipping")
        return {"series_id": series_id, "model": model, "horizon": h,
                "forecast": forecast.tolist(), "total_expected_demand": float(forecast.sum()),
                "metadata": {"source": "trusted_saved_ml", "forecast_origin_day": origin,
                             "trained_through_day": snapshot["trained_through_day"],
                             "feature_pipeline_id": snapshot["feature_pipeline_id"],
                             "artifact_paths": files}}
    except ToolError:
        raise
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ToolError(f"Saved ML inference failed: {type(exc).__name__}: {exc}") from exc


def route_with_saved_ml(
    series_id: str, history: list[float], origin: int, horizon: int,
    registry_path: str | Path | None, feature_provider: FeatureProvider | None,
    alpha: float = 0.2, n_folds: int = 3, min_train_days: int = 90,
    generalizable_registry_path: str | Path | None = None,
    nhits_scalable_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare statistical and available ML on precisely matched historical windows.

    ML is eligible only with the exact-origin snapshots for ALL folds and final origin.
    Missing/invalid ML is excluded with a reason; statistical routing still works.
    """
    history = _history(history)
    statistical = select_by_rolling_validation(series_id, history.tolist(), horizon, alpha, n_folds, min_train_days)
    cutoffs = validation_cutoffs(len(history), horizon, n_folds, min_train_days)
    if not cutoffs:
        return {**statistical, "ml_excluded": {name: "Insufficient history for matched validation" for name in ML_NAMES},
                "reason": statistical["reason"] + " Saved ML was not evaluated."}
    abs_cutoffs = [origin - (len(history) - cutoff) for cutoff in cutoffs]
    if abs_cutoffs[0] < 1:
        raise ToolError("Historical origin mapping invalid")
    means = dict(statistical["validation_mae"])
    errors = dict(statistical.get("fold_mae", {}))
    excluded = {}
    generalizable_excluded = {}
    nhits_excluded = {}
    if registry_path is None or feature_provider is None:
        excluded = {name: "No trusted registry and as-of feature provider supplied" for name in ML_NAMES}
    else:
        for name in ML_NAMES:
            try:
                # Verify that a final-origin snapshot is ready BEFORE ranking the model.
                doc, _ = _registry(registry_path)
                _snapshot(doc, name, origin)
                fold_scores = []
                for cutoff, abs_origin in zip(cutoffs, abs_cutoffs):
                    pred = predict_snapshot(series_id, history[:cutoff].tolist(), abs_origin,
                                            horizon, name, registry_path, feature_provider)
                    truth = history[cutoff:cutoff + horizon]
                    fold_scores.append(float(np.abs(np.asarray(pred["forecast"]) - truth).mean()))
                if len(fold_scores) != len(cutoffs):
                    raise ToolError("Not all required historical folds completed")
                means[name] = float(np.mean(fold_scores))
                errors[name] = fold_scores
            except (ToolError, KeyError, TypeError, ValueError) as exc:
                excluded[name] = str(exc)
    from .generalizable_routing import NAMES as GENERAL_NAMES, predict_generalizable_snapshot
    for name in GENERAL_NAMES:
        if generalizable_registry_path is None:
            generalizable_excluded[name] = 'No generalizable LightGBM registry supplied'
            continue
        try:
            predict_generalizable_snapshot(series_id,history.tolist(),origin,horizon,name,
                                            generalizable_registry_path)
            fold_scores=[]
            for cutoff,abs_origin in zip(cutoffs,abs_cutoffs):
                pred=predict_generalizable_snapshot(series_id,history[:cutoff].tolist(),abs_origin,
                                                     horizon,name,generalizable_registry_path)
                fold_scores.append(float(np.abs(np.asarray(pred['forecast'])-history[cutoff:cutoff+horizon]).mean()))
            means[name]=float(np.mean(fold_scores));errors[name]=fold_scores
        except (ToolError,ValueError,TypeError,KeyError,OSError) as exc:
            generalizable_excluded[name]=str(exc)
    from .nhits_inference import SCALABLE_MODEL_NAME, predict_nhits_scalable
    if nhits_scalable_root is None:
        nhits_excluded[SCALABLE_MODEL_NAME] = 'No NHITS_scalable checkpoint root supplied'
    else:
        try:
            from .nhits_inference import scalable_checkpoint_path
            scalable_checkpoint_path(nhits_scalable_root, origin)
            fold_scores = []
            for cutoff, abs_origin in zip(cutoffs, abs_cutoffs):
                prediction = predict_nhits_scalable(
                    series_id, abs_origin, history[:cutoff].tolist(), horizon,
                    str(nhits_scalable_root))
                truth = history[cutoff:cutoff + horizon]
                fold_scores.append(float(np.abs(np.asarray(prediction['forecast']) - truth).mean()))
            means[SCALABLE_MODEL_NAME] = float(np.mean(fold_scores))
            errors[SCALABLE_MODEL_NAME] = fold_scores
        except (Exception,) as exc:
            nhits_excluded[SCALABLE_MODEL_NAME] = str(exc)
    # Identical fold panel for every eligible model; deterministic tie order.
    preference = ["TSB", "SES", "SBA", "Croston", "Average", "SeasonalNaive", "Naive", *ML_NAMES, *GENERAL_NAMES, SCALABLE_MODEL_NAME]
    winner = min(means, key=lambda name: (means[name], preference.index(name)))
    return {"model": winner, "method": "historical_rolling_mae",
            "fold_cutoffs": cutoffs, "fold_origin_days": abs_cutoffs,
            "fold_mae": errors, "validation_mae": means, "ml_excluded": excluded,
            "generalizable_excluded": generalizable_excluded,
            "nhits_excluded": nhits_excluded,
            "alpha": alpha, "n_folds": len(cutoffs),
            "reason": f"{winner} selected by lowest matched historical MAE over {len(cutoffs)} folds ({means[winner]:.4f}). "
                      f"ML candidates excluded: {list(excluded)}; generalizable candidates excluded: "
                      f"{list(generalizable_excluded)}. No overall superiority claim."}
