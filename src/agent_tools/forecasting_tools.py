"""Inference-only tool boundary. Never trains models or accesses future observations.

Copy src/agent_tools into an existing project containing src/forecasting/statistical_models.py.
Saved ML artifacts must be explicitly registered; there is deliberately no automatic training.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import json
import numpy as np

STATISTICAL = ('Average','Naive','SeasonalNaive','SES','Croston','SBA','TSB')

class ToolError(ValueError):
    """Input validation or model availability failure safe to show to an agent."""


def _history(values: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ToolError('Sales history must be numeric') from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ToolError('Expected a nonempty one-dimensional sales history')
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ToolError('Historical sales must be finite and nonnegative')
    return arr


def _horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not (1 <= int(horizon) <= 365):
        raise ToolError('Forecast horizon must be an integer in [1, 365]')
    return int(horizon)


def describe_history(series_id: str, sales_history: Sequence[float]) -> dict[str, Any]:
    """A small diagnostic tool using only user-supplied historical observations."""
    if not series_id or not str(series_id).strip():
        raise ToolError('series_id is required')
    a = _history(sales_history)
    positive = a[a > 0]
    return {'series_id':str(series_id), 'n_days':int(a.size),
            'mean_daily_sales':float(a.mean()),'positive_days':int(positive.size),
            'zero_ratio':float(np.mean(a == 0)),
            'last_day_sales':float(a[-1]),
            'mean_positive_quantity':float(positive.mean()) if positive.size else None,
            'warning':'Sales do not establish stock availability or product launch date.'}


def forecast_statistical(series_id: str, sales_history: Sequence[float], model: str = 'TSB',
                         horizon: int = 28, alpha: float = .2, **model_params: Any) -> dict[str, Any]:
    """Call the project's shared statistical model implementation; never retrain ML.

    Expects src.forecasting.statistical_models.forecast_matrix(model, history_2d, horizon, **kwargs)
    as used by Notebook 2B. The output is validated rather than silently clipped.
    """
    if model not in STATISTICAL:
        raise ToolError(f'Unsupported statistical model: {model}. Supported: {STATISTICAL}')
    h = _horizon(horizon)
    a = _history(sales_history)
    if not series_id or not str(series_id).strip():
        raise ToolError('series_id is required')
    kwargs = dict(model_params)
    if model in ('SES','Croston','SBA','TSB'):
        if not isinstance(alpha, (int,float)) or isinstance(alpha,bool) or not (0 < alpha <= 1):
            raise ToolError('alpha must be in (0, 1]')
        kwargs['alpha'] = float(alpha)
    elif kwargs:
        raise ToolError('Additional parameters are not supported for this model')
    try:
        from src.forecasting.statistical_models import forecast_matrix
        result = forecast_matrix(model, a[None,:], h, **kwargs)
    except Exception as exc:
        raise ToolError(f'Statistical forecast failed: {type(exc).__name__}: {exc}') from exc
    values = np.asarray(result, dtype=float)
    if values.shape == (1,h): values = values[0]
    if values.shape != (h,) or not np.all(np.isfinite(values)) or np.any(values < -1e-9):
        raise ToolError(f'Invalid model output; expected {h} nonnegative finite values')
    values = np.maximum(values, 0.)
    return {'series_id':str(series_id),'model':model,'horizon':h,
            'forecast':[float(v) for v in values],
            'total_expected_demand':float(values.sum()),
            'metadata':{'history_days':int(a.size),'alpha':kwargs.get('alpha'),
                        'source':'src.forecasting.statistical_models.forecast_matrix',
                        'forecast_origin':'end_of_provided_history',
                        'selection_status':'model supplied by caller; not validation-selected'}}


def available_models(artifact_registry: str | Path | None = None) -> dict[str, Any]:
    """Model availability, not an assertion of model quality."""
    result = {'statistical':list(STATISTICAL),'saved_ml':{}}
    if artifact_registry is not None:
        p = Path(artifact_registry)
        if p.is_file():
            reg = json.loads(p.read_text(encoding='utf-8'))
            result['saved_ml'] = {name:(Path(v['artifact_path']) if Path(v['artifact_path']).is_absolute() else p.parent/Path(v['artifact_path'])).is_file() for name,v in reg.get('models',{}).items()}
    return result


def forecast_saved_lightgbm(series_id: str, future_features: Any, model_name: str,
                            registry_path: str | Path, horizon: int = 28) -> dict[str, Any]:
    """Predict using an explicitly registered, already-fitted LightGBM artifact.

    Registry JSON format: {"models":{"DirectLightGBM":{"artifact_path":"...",
      "feature_names":["lag_7",...],"artifact_format":"joblib","trained_through_day":1885}}}
    future_features: pandas.DataFrame with exactly horizon ordered rows, computed as-of origin.
    Rejects absent models, mismatched columns and any training inside the tool.
    Assumes joblib artifact is trusted; loading pickle/joblib files from untrusted sources is unsafe.
    """
    h = _horizon(horizon)
    import pandas as pd
    if not isinstance(future_features, pd.DataFrame) or len(future_features) != h:
        raise ToolError(f'future_features must be a DataFrame with exactly {h} future-day rows')
    rpath = Path(registry_path).resolve()
    if not rpath.is_file(): raise ToolError(f'Model registry not found: {rpath}')
    registry = json.loads(rpath.read_text(encoding='utf-8'))
    entry = registry.get('models',{}).get(model_name)
    if entry is None: raise ToolError(f'Model {model_name!r} is not registered; no retraining performed')
    if entry.get('artifact_format') != 'joblib': raise ToolError('Only explicitly trusted joblib artifacts are supported')
    path = Path(entry['artifact_path'])
    if not path.is_absolute(): path = (rpath.parent/path).resolve()
    if not path.is_file(): raise ToolError(f'Saved model artifact unavailable: {path}; no retraining performed')
    names = entry.get('feature_names')
    if not isinstance(names,list) or not names or any(x not in future_features.columns for x in names):
        raise ToolError('Feature schema mismatch between artifact and future features')
    if 'forecast_origin_day' not in entry:
        raise ToolError('Registry lacks forecast_origin_day needed for leakage checks')
    if int(entry['trained_through_day']) > int(entry['forecast_origin_day']):
        raise ToolError('Artifact training cutoff exceeds forecast origin')
    import joblib
    model = joblib.load(path)  # ONLY trusted local artifacts
    values = np.asarray(model.predict(future_features[names]), dtype=float)
    if values.shape != (h,) or not np.isfinite(values).all():
        raise ToolError('Saved ML model returned invalid prediction values')
    if np.any(values < 0): raise ToolError('ML output contains negative predictions; configure validated postprocessing explicitly')
    return {'series_id':str(series_id),'model':model_name,'horizon':h,
            'forecast':values.tolist(),'total_expected_demand':float(values.sum()),
            'metadata':{'artifact_path':str(path),'training_cutoff':entry['trained_through_day'],
                        'forecast_origin_day':entry['forecast_origin_day'],
                        'source':'saved_model_inference','selection_status':'caller-provided artifact'}}
