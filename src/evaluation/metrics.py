"""Forecast metrics with explicit undefined-value handling."""
from __future__ import annotations
import numpy as np


def _pair(actual, prediction):
    a, p = np.asarray(actual, float), np.asarray(prediction, float)
    if a.shape != p.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("actual and prediction must be aligned, non-empty, and finite")
    return a, p


def mae(actual, prediction):
    a, p = _pair(actual, prediction); return float(np.mean(np.abs(p-a)))


def mean_signed_bias(actual, prediction):
    a, p = _pair(actual, prediction); return float(np.mean(p-a))


def wape(actual, prediction):
    a, p = _pair(actual, prediction); d = np.abs(a).sum()
    return float(np.abs(p-a).sum()/d) if d > 0 else np.nan


def _scale(train, squared=False, seasonality=1):
    y = np.asarray(train, float); m = int(seasonality)
    if y.ndim != 1 or y.size <= m or m < 1: return np.nan
    differences = y[m:] - y[:-m]
    return float(np.mean(differences**2 if squared else np.abs(differences)))


def mase(actual, prediction, train, seasonality=1):
    scale = _scale(train, False, seasonality)
    return mae(actual, prediction)/scale if np.isfinite(scale) and scale > 0 else np.nan


def rmsse(actual, prediction, train, seasonality=1):
    a, p = _pair(actual, prediction); scale = _scale(train, True, seasonality)
    return float(np.sqrt(np.mean((p-a)**2)/scale)) if np.isfinite(scale) and scale > 0 else np.nan


def underforecast_quantity(actual, prediction):
    a, p = _pair(actual, prediction); return float(np.maximum(a-p, 0).sum())


def overforecast_quantity(actual, prediction):
    a, p = _pair(actual, prediction); return float(np.maximum(p-a, 0).sum())


def inventory_cost_proxy(actual, prediction, underforecast_penalty=5.0, overforecast_penalty=1.0):
    if underforecast_penalty < 0 or overforecast_penalty < 0: raise ValueError("penalties must be non-negative")
    return underforecast_penalty*underforecast_quantity(actual,prediction)+overforecast_penalty*overforecast_quantity(actual,prediction)

def series_metric_records(actual,predictions,training,series_ids,demand_types,*,underforecast_penalty=5.,overforecast_penalty=1.):
    """Return one standardized metric row per series and model."""
    import pandas as pd
    actual=np.asarray(actual,float); training=np.asarray(training,float)
    rows=[]
    for model,forecast in predictions.items():
        forecast=np.asarray(forecast,float)
        if forecast.shape!=actual.shape: raise ValueError(f"{model} forecast is not aligned with actual")
        for i,(series_id,demand_type) in enumerate(zip(series_ids,demand_types)):
            error=forecast[i]-actual[i]
            rows.append({"series_id":series_id,"demand_type":demand_type,"model":model,
                         "MAE":mae(actual[i],forecast[i]),"MASE":mase(actual[i],forecast[i],training[i]),
                         "RMSSE":rmsse(actual[i],forecast[i],training[i]),"abs_error_sum":float(np.abs(error).sum()),
                         "actual_sum":float(actual[i].sum()),"bias":mean_signed_bias(actual[i],forecast[i]),
                         "cost_proxy":inventory_cost_proxy(actual[i],forecast[i],underforecast_penalty,overforecast_penalty)})
    return pd.DataFrame(rows)

def aggregate_metric_records(group):
    """Aggregate series rows; WAPE is pooled, never an average of ratios."""
    import pandas as pd
    total=float(group["actual_sum"].sum())
    return pd.Series({"series":len(group),"MAE":group["MAE"].mean(),"MASE":group["MASE"].mean(),
                      "valid_MASE":group["MASE"].count(),"RMSSE":group["RMSSE"].mean(),
                      "valid_RMSSE":group["RMSSE"].count(),
                      "WAPE_pct":100*group["abs_error_sum"].sum()/total if total>0 else np.nan,
                      "bias":group["bias"].mean(),"total_cost_proxy":group["cost_proxy"].sum()})
