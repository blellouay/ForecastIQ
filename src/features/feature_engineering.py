"""Fixed-origin features using only observations before the origin."""
from __future__ import annotations
import numpy as np
import pandas as pd


def historical_snapshot(history, lags=(1,7,14,28,56), windows=(7,28,56)):
    y=np.asarray(history,float)
    if y.ndim != 2 or not np.isfinite(y).all() or (y<0).any(): raise ValueError("history must be a finite non-negative matrix")
    required=max((*lags,*windows))
    if y.shape[1] < required: raise ValueError("insufficient history for requested features")
    result={}
    for lag in lags: result[f"lag_{lag}"]=y[:,-lag].astype(np.float32)
    for window in windows:
        block=y[:,-window:]
        result[f"mean_{window}"]=block.mean(1).astype(np.float32)
        result[f"std_{window}"]=block.std(1).astype(np.float32)
        result[f"zero_ratio_{window}"]=(block==0).mean(1).astype(np.float32)
    if y.shape[1] >= 56:
        result["last_28_vs_prev_28"]=(y[:,-28:].mean(1)-y[:,-56:-28].mean(1)).astype(np.float32)
    return result


def fixed_origin_frame(panel, origin, horizon, *, dates=None, include_target=False):
    y=np.asarray(panel,float)
    if origin < 1 or origin+horizon > y.shape[1]: raise ValueError("invalid origin/horizon")
    snapshot=historical_snapshot(y[:,:origin]); n=y.shape[0]
    rows={k:np.repeat(v,horizon) for k,v in snapshot.items()}
    rows["horizon_step"]=np.tile(np.arange(1,horizon+1),n)
    if dates is not None:
        future=pd.DatetimeIndex(dates[origin:origin+horizon])
        rows["weekday"]=np.tile(future.dayofweek.to_numpy(),n)
        rows["month"]=np.tile(future.month.to_numpy(),n)
    frame=pd.DataFrame(rows)
    if include_target: frame["target"]=y[:,origin:origin+horizon].reshape(-1)
    return frame
