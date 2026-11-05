"""Pure, tested helpers for the ForecastIQ Streamlit dashboard.

No model training and no network calls are performed here.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import pandas as pd

_M5_ID = re.compile(r"^(FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+_(CA|TX|WI)_\d+_(validation|evaluation)$", re.I)
_MODELS = ('auto','Average','Naive','SeasonalNaive','SES','Croston','SBA','TSB','DirectLightGBM','HurdleLightGBM','GeneralizableDirectLightGBM','GeneralizableHurdleLightGBM','NHITS_small','NHITS_scalable','DeepAR')


def parse_m5_id(series_id: str) -> tuple[str, str]:
    """Return (item, store) from an official M5 row identifier."""
    match = _M5_ID.fullmatch(str(series_id).strip())
    if match is None:
        raise ValueError(f'Invalid M5 identifier: {series_id!r}')
    parts = series_id.rsplit('_', 3)
    # tail is store region + store number + split e.g CA_1_validation
    return '_'.join(parts[:-3]), parts[-3] + '_' + parts[-2]


def make_request(item: str, store: str, model: str, horizon: int = 28) -> str:
    if not re.fullmatch(r'(FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+', item, re.I):
        raise ValueError('Invalid M5 product ID')
    if not re.fullmatch(r'(CA|TX|WI)_\d+', store, re.I):
        raise ValueError('Invalid M5 store ID')
    if model not in _MODELS:
        raise ValueError('Unsupported forecast model')
    if horizon != 28:
        raise ValueError('Only the validated 28-day horizon is supported')
    return f'Forecast {item.upper()} in store {store.upper()} for 28 days using {model}'


def read_catalog(csv_path: str) -> dict[str, list[str]]:
    """Read only the M5 ID column (avoids loading the full wide dataset)."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'M5 CSV not found: {path}')
    data = pd.read_csv(path, usecols=['id'], dtype={'id': str})
    catalog: dict[str, set[str]] = {}
    for identifier in data['id'].dropna():
        try:
            item, store = parse_m5_id(identifier)
        except ValueError:
            continue
        catalog.setdefault(item, set()).add(store)
    if not catalog:
        raise ValueError('No valid M5 product-store IDs found in dataset')
    return {item: sorted(stores) for item, stores in sorted(catalog.items())}


def filter_catalog(catalog: dict[str, list[str]], mode: str,
                   mapping_path: str = 'models/lightgbm/series_mapping.json') -> dict[str, list[str]]:
    if mode in ('All products', 'N-HiTS-compatible products'):
        return catalog
    if mode != 'LightGBM-compatible products':
        raise ValueError(f'Unknown product filter: {mode}')
    import json
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    supported = set(mapping.get('series_id_to_code', {}))
    result = {}
    for item, stores in catalog.items():
        kept = [store for store in stores if f'{item}__{store}' in supported]
        if kept:
            result[item] = kept
    return result


def read_history(csv_path: str, series_id: str, origin: int, days: int = 90) -> pd.DataFrame:
    """Read historical sales for charting only; never read target days after the origin."""
    if origin < 1 or days < 1:
        raise ValueError('origin and days must be positive')
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    selected_days = [f'd_{day}' for day in range(max(1, origin-days+1), origin+1)]
    missing = [day for day in selected_days if day not in header]
    if missing:
        raise ValueError(f'Missing required historical day columns (first: {missing[0]})')
    subset = pd.read_csv(path, usecols=['id', *selected_days])
    matches = subset.loc[subset['id'] == series_id]
    if len(matches) != 1:
        raise ValueError(f'Expected one row for {series_id}, found {len(matches)}')
    values = matches.iloc[0]
    return pd.DataFrame({'day': list(range(max(1, origin-days+1), origin+1)),
                         'actual': [float(values[d]) for d in selected_days]})


def forecast_table(result: dict[str, Any], origin: int) -> pd.DataFrame:
    """Normalize verified model output; no predictions are synthesized."""
    if result.get('status') != 'completed':
        raise ValueError(result.get('error', 'Forecast did not complete'))
    payload = result.get('forecast_result') or {}
    values = payload.get('forecast')
    if not isinstance(values, (list, tuple)) or len(values) != 28:
        raise ValueError('Expected 28 model-generated predictions')
    import math
    if not all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Forecast contains invalid or negative predictions')
    return pd.DataFrame({'day': list(range(origin+1, origin+29)),
                         'forecast': [float(v) for v in values]})


def selection_frame(result: dict[str, Any]) -> pd.DataFrame:
    evidence = result.get('selection_evidence') or {}
    scores = evidence.get('validation_mae') or {}
    rows = []
    for model, score in scores.items():
        if isinstance(score, (int, float)):
            rows.append({'Model': str(model), 'Historical validation MAE': float(score)})
    return pd.DataFrame(rows).sort_values('Historical validation MAE') if rows else pd.DataFrame()


def model_availability_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = result.get('model_availability') or []
    columns = ['Model','Family','Inference','Auto selection','Reason']
    normalized = [{
        'Model': row.get('model'), 'Family': row.get('family'),
        'Inference': 'Yes' if row.get('inference_eligible') else 'No',
        'Auto selection': 'Yes' if row.get('selection_eligible') else 'No',
        'Reason': row.get('reason',''),
    } for row in rows]
    return pd.DataFrame(normalized, columns=columns)


def extract_ai_part(response: str, result: dict[str, Any]) -> tuple[str, str]:
    """Read-only display of optional LLM commentary with explicit status."""
    if isinstance(result.get('ai_explanation'), str) and result['ai_explanation'].strip():
        return result['ai_explanation'].strip(), str(result.get('llm_status', 'success'))
    marker = 'AI interpretation (non-numeric):'
    if marker in response:
        return response.split(marker, 1)[1].strip(), 'success'
    return '', str(result.get('llm_status') or 'offline_or_unavailable')
