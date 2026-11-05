"""Transparent scenario detection and comparable-product cold-start baseline."""
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd

MIN_GENERALIZABLE_HISTORY=56
SHORT_HISTORY_DAYS=56

def detect_forecasting_scenario(series_id,history,mapping_path='models/lightgbm/series_mapping.json'):
    values=np.asarray(history,dtype=float)
    if values.ndim!=1:raise ValueError('history must be one dimensional')
    if len(values)==0:
        return {'scenario':'no_history_cold_start','historical_observations':0,
                'warning':'No sales observations were supplied; zeros before first sale are not assumed.'}
    if not np.isfinite(values).all():
        return {'scenario':'missing_history','historical_observations':len(values),
                'warning':'Missing observations must be resolved before forecasting.'}
    if len(values)<SHORT_HISTORY_DAYS:
        return {'scenario':'short_history','historical_observations':len(values),
                'warning':f'Fewer than {SHORT_HISTORY_DAYS} observations; lag-56 global models are unavailable.'}
    known=False
    try:
        from src.features.m5_lightgbm import normalize_series_id
        mapping=json.loads(Path(mapping_path).read_text(encoding='utf-8'))
        known=normalize_series_id(series_id) in mapping.get('series_id_to_code',{})
    except Exception:pass
    return {'scenario':'known_series' if known else 'unseen_series_with_history',
            'historical_observations':len(values),'warning':'Stock availability and true launch date are unknown.'}


def comparable_product_forecast(series_id,metadata,sales_path,origin,horizon=28,min_comparables=3):
    if horizon!=28:raise ValueError('Cold-start baseline supports a 28-day horizon')
    required={'category','department','store'}
    if not isinstance(metadata,dict) or not required.issubset(metadata):
        raise ValueError(f'Cold-start metadata requires {sorted(required)}')
    day_cols=[f'd_{i}' for i in range(1,origin+1)]
    header=pd.read_csv(sales_path,nrows=0).columns
    id_col='id' if 'id' in header else 'series_id' if 'series_id' in header else None
    if id_col is None:raise ValueError('Comparable dataset requires id or series_id')
    missing=[c for c in day_cols if c not in header]
    if missing:raise ValueError(f'Comparable dataset lacks {missing[0]}')
    exact_parts=[];broad_parts=[]
    for raw in pd.read_csv(sales_path,usecols=[id_col,*day_cols[-28:]],chunksize=1000):
        ids=raw[id_col].astype(str).str.replace(r'_(validation|evaluation)$','',regex=True)
        ids=ids.str.replace(r'_((?:CA|TX|WI)_\d+)$',r'__\1',regex=True)
        item=ids.str.split('__').str[0];store=ids.str.split('__').str[1]
        department=item.str.split('_').str[:2].str.join('_');category=item.str.split('_').str[0]
        values=raw[day_cols[-28:]].apply(pd.to_numeric,errors='coerce')
        if values.isna().any().any():raise ValueError('Comparable sales contain missing observations')
        per_series=values.mean(axis=1)
        exact=(category==metadata['category'])&(department==metadata['department'])&(store==metadata['store'])
        broad=category==metadata['category']
        exact_parts.extend(per_series[exact].tolist());broad_parts.extend(per_series[broad].tolist())
    exact_values=np.asarray(exact_parts,float);broad_values=np.asarray(broad_parts,float)
    if len(broad_values)<min_comparables:raise ValueError('Insufficient comparable products for cold-start baseline')
    broad_median=float(np.median(broad_values));n=len(exact_values)
    exact_median=float(np.median(exact_values)) if n else broad_median
    weight=n/(n+5.0);estimate=max(0.0,weight*exact_median+(1-weight)*broad_median)
    forecast=[estimate]*horizon
    return {'series_id':series_id,'model':'ColdStartComparable','horizon':horizon,
            'forecast':forecast,'total_expected_demand':float(sum(forecast)),'metadata':{
                'scenario':'simulated_no_history','comparable_products':n,
                'broader_category_products':len(broad_values),'exact_group_median':exact_median,
                'category_median':broad_median,'shrinkage_weight':weight,
                'history_cutoff_day':origin,
                'warning':'M5 has no verified launch dates or complete attributes; this is a simulated no-history baseline.'}}
