"""Hurdle LightGBM: P(Y>0) times E[Y | Y>0]."""
from __future__ import annotations
import numpy as np
from .lightgbm_direct import fit_predict_frames

def fit_hurdle(X,y,params=None,categorical_features=()):
    try: import lightgbm as lgb
    except ImportError as exc: raise ImportError("install lightgbm to use fit_hurdle") from exc
    y=np.asarray(y,float); positive=y>0
    if positive.sum()==0 or positive.all(): raise ValueError("hurdle training requires zero and positive targets")
    defaults=dict(n_estimators=250,learning_rate=.05,num_leaves=31,random_state=42,n_jobs=-1,verbosity=-1)
    defaults.update(params or {})
    categorical=list(categorical_features)
    classifier=lgb.LGBMClassifier(objective="binary",**defaults).fit(X,positive.astype("int8"),categorical_feature=categorical)
    quantity=lgb.LGBMRegressor(objective="regression",**defaults).fit(X.loc[positive],y[positive],categorical_feature=categorical)
    return classifier,quantity

def predict_hurdle(models,X):
    classifier,quantity=models
    probability=np.clip(classifier.predict_proba(X)[:,1],0,1)
    conditional=np.clip(quantity.predict(X),0,None)
    return probability*conditional,probability,conditional

def fit_predict_hurdle_and_direct_frames(X_train,y_train,X_future,*,params=None,categorical_features=()):
    """Fit hurdle and direct models to identical prepared rows and predict both."""
    y=np.asarray(y_train,float); positive=y>0
    hurdle_models=fit_hurdle(X_train,y,params,categorical_features)
    hurdle,probability,conditional=predict_hurdle(hurdle_models,X_future)
    direct,direct_model=fit_predict_frames(X_train,y,X_future,params=params,categorical_features=categorical_features)
    classifier,quantity=hurdle_models
    return {"hurdle":hurdle,"direct":direct,"probability":probability,"conditional":conditional,
            "classifier":classifier,"quantity_model":quantity,"direct_model":direct_model,
            "train_rows":len(y),"positive_train_rows":int(positive.sum())}
