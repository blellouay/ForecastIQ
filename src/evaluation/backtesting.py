"""Chronological multi-origin backtesting primitives."""
from __future__ import annotations
import numpy as np


def expanding_origins(length, horizon=28, n_origins=6, *, final_end=None, non_overlapping=True, min_history=112):
    end = length if final_end is None else int(final_end)
    step = horizon if non_overlapping else 1
    origins = list(range(end-horizon, min_history-1, -step))[:int(n_origins)]
    if len(origins) < n_origins: raise ValueError("insufficient history for requested origins")
    return sorted(origins)


def backtest_series(values, predictor, origins, horizon=28):
    y=np.asarray(values,float); records=[]
    for origin in origins:
        if origin < 1 or origin+horizon > y.size: raise ValueError("invalid forecast origin")
        prediction=np.asarray(predictor(y[:origin],horizon),float)
        if prediction.shape != (horizon,): raise ValueError("predictor returned wrong horizon")
        for step,(actual,pred) in enumerate(zip(y[origin:origin+horizon],prediction),1):
            records.append({"forecast_origin":origin,"target_day":origin+step,
                            "horizon_step":step,"actual":actual,"prediction":pred})
    return records
