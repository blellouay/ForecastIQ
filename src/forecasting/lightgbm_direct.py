"""Optional LightGBM direct forecaster built on the shared feature pipeline."""
from __future__ import annotations
import numpy as np
from src.features.feature_engineering import fixed_origin_frame

def fit_predict_frames(X_train, y_train, X_future, *, params=None, categorical_features=()):
    """Fit one direct LightGBM model on prepared, origin-safe feature frames."""
    try: import lightgbm as lgb
    except ImportError as exc: raise ImportError("install lightgbm to use direct forecasting") from exc
    defaults=dict(n_estimators=250,learning_rate=.05,num_leaves=31,min_child_samples=70,
                  colsample_bytree=.9,subsample=.9,subsample_freq=1,reg_lambda=1.,
                  random_state=42,n_jobs=-1,verbosity=-1)
    defaults.update(params or {})
    model=lgb.LGBMRegressor(objective="regression",**defaults)
    model.fit(X_train,np.asarray(y_train,float),categorical_feature=list(categorical_features))
    prediction=np.clip(model.predict(X_future),0,None)
    return prediction,model

def fit_direct(panel, origins, *, horizon=28, dates=None, feature_set="historical_weekday_month", params=None):
    try: import lightgbm as lgb
    except ImportError as exc: raise ImportError("install lightgbm to use fit_direct") from exc
    frames=[fixed_origin_frame(panel,o,horizon,dates=dates,include_target=True) for o in origins]
    import pandas as pd
    data=pd.concat(frames,ignore_index=True); y=data.pop("target")
    allowed={"historical":[],"historical_weekday":["weekday"],"historical_month":["month"],"historical_weekday_month":["weekday","month"]}
    if feature_set not in allowed: raise ValueError("unknown feature_set")
    calendar={"weekday","month"}; columns=[c for c in data if c not in calendar or c in allowed[feature_set]]
    defaults=dict(n_estimators=250,learning_rate=.05,num_leaves=31,random_state=42,n_jobs=-1,verbosity=-1)
    defaults.update(params or {})
    return lgb.LGBMRegressor(objective="regression",**defaults).fit(data[columns],y),columns

def predict_direct(model, columns, panel, origin, *, horizon=28, dates=None):
    frame=fixed_origin_frame(panel,origin,horizon,dates=dates)
    return np.clip(model.predict(frame[columns]),0,None).reshape(len(panel),horizon)
