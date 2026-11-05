"""Separate global LightGBM family that supports unseen series with history."""
from __future__ import annotations
import joblib
import numpy as np
from pathlib import Path
from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames


def fit_generalizable(X,y,*,params,categorical_features):
    return fit_predict_hurdle_and_direct_frames(
        X,np.asarray(y,float),X.iloc[:1].copy(),params=params,
        categorical_features=categorical_features)


def predict_generalizable(frame, model_dir: str|Path, origin: int, kind: str):
    root=Path(model_dir)
    if kind=='direct':
        model=joblib.load(root/f'direct_origin_{origin}.joblib')
        values=np.clip(model.predict(frame),0,None)
    elif kind=='hurdle':
        classifier=joblib.load(root/f'hurdle_classifier_origin_{origin}.joblib')
        regressor=joblib.load(root/f'hurdle_regressor_origin_{origin}.joblib')
        classes=list(classifier.classes_)
        p=classifier.predict_proba(frame)[:,classes.index(1)]
        values=np.clip(p,0,1)*np.clip(regressor.predict(frame),0,None)
    else: raise ValueError('kind must be direct or hurdle')
    if values.shape!=(28,) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Invalid generalizable LightGBM forecast')
    return values
