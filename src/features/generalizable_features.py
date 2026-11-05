"""Origin-safe features for global models without product-specific identifiers."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'horizon','store','department','category',
    'lag_1','lag_7','lag_14','lag_28','lag_56',
    'mean_7','std_7','zero_ratio_7','mean_14','std_14','zero_ratio_14',
    'mean_28','std_28','zero_ratio_28','positive_mean','recent_trend',
    'adi','cv2','demand_type','day_of_week','month',
]
CATEGORICAL_FEATURES = ['store','department','category','demand_type']
PIPELINE_ID = 'm5-generalizable-historical-v1'
_ID = re.compile(r'^(?P<category>FOODS|HOBBIES|HOUSEHOLD)_(?P<dept>\d+)_(?P<item>\d+)__(?P<store>(?:CA|TX|WI)_\d+)$')


def canonical_metadata(series_id: str) -> dict[str,str]:
    raw = str(series_id)
    for suffix in ('_validation','_evaluation'):
        if raw.endswith(suffix): raw = raw[:-len(suffix)]
    if '__' not in raw:
        match = re.fullmatch(r'(.+)_((?:CA|TX|WI)_\d+)', raw)
        if not match: raise ValueError(f'Invalid M5 series ID: {series_id!r}')
        raw = match.group(1)+'__'+match.group(2)
    match = _ID.fullmatch(raw)
    if not match: raise ValueError(f'Invalid M5 canonical identity: {raw!r}')
    return {'series_id':raw, 'store':match.group('store'),
            'department':f"{match.group('category')}_{match.group('dept')}",
            'category':match.group('category')}


def history_profile(history) -> dict[str,float|str]:
    y=np.asarray(history,dtype=float)
    if y.ndim!=1 or len(y)<56 or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Generalizable features require at least 56 finite nonnegative observations')
    positive=y[y>0]; intervals=np.diff(np.flatnonzero(y>0))
    adi=float(len(y)/len(positive)) if len(positive) else float(len(y)+1)
    cv2=float((positive.std()/positive.mean())**2) if len(positive)>1 and positive.mean()>0 else 0.0
    demand_type=('Smooth' if adi<1.32 and cv2<.49 else 'Erratic' if adi<1.32 else
                 'Intermittent' if cv2<.49 else 'Lumpy')
    result={f'lag_{lag}':float(y[-lag]) for lag in (1,7,14,28,56)}
    for window in (7,14,28):
        block=y[-window:]
        result.update({f'mean_{window}':float(block.mean()),f'std_{window}':float(block.std()),
                       f'zero_ratio_{window}':float((block==0).mean())})
    result.update(positive_mean=float(positive.mean()) if len(positive) else 0.0,
                  recent_trend=float(y[-14:].mean()-y[-28:-14].mean()),
                  adi=adi,cv2=cv2,demand_type=demand_type)
    return result


def feature_frame(series_ids, histories, origin: int, horizon: int, dates,
                  categories: dict[str,list[str]]|None=None) -> pd.DataFrame:
    if horizon!=28: raise ValueError('Generalizable LightGBM supports a 28-day horizon')
    future=pd.DatetimeIndex(dates[origin:origin+horizon])
    frames=[]
    for series_id,history in zip(series_ids,histories):
        meta=canonical_metadata(series_id); profile=history_profile(history)
        rows={k:np.repeat(v,horizon) for k,v in {**meta,**profile}.items() if k!='series_id'}
        rows['horizon']=np.arange(1,horizon+1,dtype=np.int16)
        rows['day_of_week']=future.dayofweek.to_numpy(dtype=np.int8)
        rows['month']=future.month.to_numpy(dtype=np.int8)
        frames.append(pd.DataFrame(rows)[FEATURE_NAMES])
    frame=pd.concat(frames,ignore_index=True)
    for column in CATEGORICAL_FEATURES:
        values=categories[column] if categories else sorted(frame[column].astype(str).unique())
        frame[column]=pd.Categorical(frame[column],categories=values)
        if frame[column].isna().any(): raise ValueError(f'Unsupported {column} category')
    return frame
