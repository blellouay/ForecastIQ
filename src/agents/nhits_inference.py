"""Inference-only adapters for saved NeuralForecast N-HiTS checkpoints.

The saved weights were trained through M5 day 1857. They may consume supplied
history at later origins, but do not provide leakage-safe snapshots for earlier
historical auto-selection folds.
"""
from __future__ import annotations
from functools import lru_cache
from datetime import date, timedelta
from pathlib import Path
import math

MODEL_NAME = 'NHITS_small'
SCALABLE_MODEL_NAME = 'NHITS_scalable'
SCALABLE_ORIGINS = (1801, 1829, 1857, 1885)
SCALABLE_ROOT = 'models/nhits_full_m5/full_all'
TRAINING_CUTOFF_DAY = 1857
M5_FIRST_DATE = date(2011, 1, 29)

class NHITSInferenceError(ValueError):
    pass


def _canonical(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier.strip():
        raise NHITSInferenceError('series_id is required')
    raw = identifier.strip()
    for suffix in ('_validation', '_evaluation'):
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
            break
    if '__' in raw:
        return raw
    # Only canonicalize a standard M5 item_store ID with store in CA_1..4, TX_1..3, WI_1..3.
    import re
    match = re.fullmatch(r'(.+)_(CA_[1-4]|TX_[1-3]|WI_[1-3])', raw)
    if not match:
        raise NHITSInferenceError(f'Unrecognized M5 series identifier: {identifier!r}')
    return match.group(1) + '__' + match.group(2)


@lru_cache(maxsize=2)
def _load(path: str):
    try:
        from neuralforecast import NeuralForecast
    except ImportError as exc:
        raise NHITSInferenceError('Install neuralforecast in the project .venv') from exc
    root = Path(path)
    needed = ['configuration.json','configuration.safetensors','dataset.json','dataset.safetensors','NHITS_small_0.safetensors']
    missing = [name for name in needed if not (root/name).is_file()]
    if missing:
        raise NHITSInferenceError(f'Incomplete N-HiTS checkpoint, missing: {missing}')
    nf = NeuralForecast.load(str(root))
    if len(nf.models) != 1 or getattr(nf.models[0], 'alias', None) != MODEL_NAME:
        raise NHITSInferenceError('Checkpoint does not contain only NHITS_small')
    for model in nf.models:
        model.trainer_kwargs.update({'accelerator':'cpu','devices':1,'strategy':'auto'})
        model.to('cpu')
    return nf


def _metadata(path: str | Path, expected_origin: int) -> dict:
    import json
    root = Path(path)
    required = ('configuration.json', 'configuration.safetensors', 'dataset.json',
                'dataset.safetensors', f'{SCALABLE_MODEL_NAME}_0.safetensors',
                'retsci_metadata.json')
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise NHITSInferenceError(f'Incomplete scalable N-HiTS checkpoint, missing: {missing}')
    try:
        metadata = json.loads((root / 'retsci_metadata.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise NHITSInferenceError('Invalid scalable N-HiTS provenance metadata') from exc
    expected = {'model': 'NHITS', 'alias': SCALABLE_MODEL_NAME, 'run_size': 'full_all',
                'smoke_test': False, 'training_cutoff': expected_origin,
                'forecast_horizon': 28, 'train_series_count': 30490,
                'input_size': 56, 'max_steps': 500}
    conflicts = {key: (metadata.get(key), value) for key, value in expected.items()
                 if metadata.get(key) != value}
    if conflicts:
        raise NHITSInferenceError(f'Scalable N-HiTS provenance mismatch: {conflicts}')
    return metadata


@lru_cache(maxsize=1)
def _load_scalable(path: str, expected_origin: int):
    _metadata(path, expected_origin)
    try:
        from neuralforecast import NeuralForecast
    except ImportError as exc:
        raise NHITSInferenceError('Install neuralforecast in the project .venv') from exc
    nf = NeuralForecast.load(path)
    if len(nf.models) != 1 or getattr(nf.models[0], 'alias', None) != SCALABLE_MODEL_NAME:
        raise NHITSInferenceError('Checkpoint does not contain only NHITS_scalable')
    for model in nf.models:
        model.trainer_kwargs.update({'accelerator': 'cpu', 'devices': 1, 'strategy': 'auto'})
        model.to('cpu')
    return nf


def scalable_checkpoint_path(root: str | Path, origin_day: int) -> Path:
    if origin_day not in SCALABLE_ORIGINS:
        raise NHITSInferenceError(
            f'No NHITS_scalable snapshot for origin d_{origin_day}; available: {list(SCALABLE_ORIGINS)}')
    path = Path(root) / f'origin_{origin_day}'
    _metadata(path, origin_day)
    return path


@lru_cache(maxsize=2)
def _predict_all(path: str):
    nf = _load(path)
    pred = nf.predict()  # Uses checkpoint's stored dataset, NOT newly supplied history.
    if not {'unique_id', 'ds', MODEL_NAME}.issubset(pred.columns):
        raise NHITSInferenceError('Saved model produced an unexpected prediction schema')
    if pred.duplicated(['unique_id','ds']).any():
        raise NHITSInferenceError('Checkpoint produced duplicate series/date predictions')
    return pred


def _supplied_frame(series_id: str, historical_sales, origin_day: int):
    import numpy as np
    import pandas as pd
    values = np.asarray(historical_sales, dtype=float)
    if values.ndim != 1 or len(values) != origin_day:
        raise NHITSInferenceError('Supplied history must contain exactly d_1 through the requested origin')
    if not np.isfinite(values).all() or (values < 0).any():
        raise NHITSInferenceError('Supplied history must be finite and nonnegative')
    dates = pd.date_range(M5_FIRST_DATE, periods=origin_day, freq='D')
    return pd.DataFrame({'unique_id': _canonical(series_id), 'ds': dates, 'y': values})


def predict_nhits(series_id: str, origin_day: int, horizon: int = 28,
                  checkpoint_path: str = 'models/neuralforecast/nhits_small',
                  historical_sales=None) -> dict:
    """Return checkpoint-specific predictions if series, origin, and dates align.

    The snapshot embeds historical series. It must NOT be used for an arbitrary
    request date or interpreted as a forecast using the caller's latest history.
    """
    if type(origin_day) is not int or origin_day <= 0:
        raise NHITSInferenceError('forecast_origin_day is required and must be a positive integer')
    if type(horizon) is not int or horizon != 28:
        raise NHITSInferenceError('This checkpoint only supports a 28-day horizon')
    canonical_id = _canonical(series_id)
    resolved = str(Path(checkpoint_path).resolve())
    if historical_sales is None:
        df = _predict_all(resolved)
        training_ids = {str(value) for value in df['unique_id'].unique()}
        history_source = 'saved NeuralForecast checkpoint dataset'
        historical_observations = TRAINING_CUTOFF_DAY
    else:
        if origin_day < TRAINING_CUTOFF_DAY:
            raise NHITSInferenceError(
                f'Checkpoint weights were trained through d_{TRAINING_CUTOFF_DAY}; '
                f'they are not historically valid at origin d_{origin_day}')
        history = _supplied_frame(series_id, historical_sales, origin_day)
        nf = _load(resolved)
        training_ids = {str(value) for value in getattr(nf, 'uids', [])}
        df = nf.predict(df=history)
        history_source = 'caller-supplied origin-truncated M5 history'
        historical_observations = len(history)
    series_in_training = canonical_id in training_ids
    selected = df.loc[df['unique_id'] == canonical_id].sort_values('ds')
    if len(selected) != horizon:
        raise NHITSInferenceError(f'{canonical_id!r} is not supported by the saved N-HiTS checkpoint')
    expected_dates = [M5_FIRST_DATE + timedelta(days=origin_day + step - 1)
                      for step in range(1, horizon + 1)]
    actual_dates = [d.date() for d in __import__('pandas').to_datetime(selected['ds'])]
    if actual_dates != expected_dates:
        raise NHITSInferenceError(
            f'Checkpoint forecasts {actual_dates[0]} through {actual_dates[-1]}, '
            f'not the requested M5 origin d_{origin_day}. '
            'Use the matching saved checkpoint; do not relabel dates.')
    values = [float(x) for x in selected[MODEL_NAME]]
    if any(not math.isfinite(x) for x in values):
        raise NHITSInferenceError('N-HiTS returned nonfinite predictions')
    # Retail demand is nonnegative. Clipping is disclosed, never hidden.
    clipped = sum(x < 0 for x in values)
    values = [max(0.0,x) for x in values]
    return {
        'series_id': series_id, 'model': MODEL_NAME, 'horizon': horizon,
        'forecast': values, 'total_expected_demand': float(sum(values)),
        'metadata': {
            'checkpoint_path': str(Path(checkpoint_path)),
            'checkpoint_identifier': 'nhits-small-neuralforecast-3.3.0-origin-1857',
            'training_cutoff_day': TRAINING_CUTOFF_DAY,
            'forecast_origin_day': origin_day, 'forecast_start_date': str(actual_dates[0]),
            'forecast_end_date': str(actual_dates[-1]), 'series_id_canonical': canonical_id,
            'inference_device': 'cpu', 'negative_predictions_clipped': clipped,
            'history_source': history_source,
            'historical_observations': historical_observations,
            'series_in_training': series_in_training,
            'supports_unseen_series_with_history': True,
            'warning': 'Checkpoint has no matched pre-origin snapshots for all required validation folds; '
                       'it is explicit-inference only and excluded from automatic ranking.'
        }
    }


def predict_nhits_scalable(series_id: str, origin_day: int, historical_sales,
                           horizon: int = 28, checkpoint_root: str = SCALABLE_ROOT) -> dict:
    """Predict with the exact full-panel snapshot for ``origin_day`` on CPU.

    Negative raw outputs are clipped to zero both in validation and final inference.
    This is the project's declared nonnegative-demand policy, not post-score cleanup.
    """
    if type(origin_day) is not int or origin_day <= 0:
        raise NHITSInferenceError('forecast_origin_day is required and must be a positive integer')
    if type(horizon) is not int or horizon != 28:
        raise NHITSInferenceError('NHITS_scalable only supports a 28-day horizon')
    history = _supplied_frame(series_id, historical_sales, origin_day)
    path = scalable_checkpoint_path(checkpoint_root, origin_day).resolve()
    nf = _load_scalable(str(path), origin_day)
    predictions = nf.predict(df=history)
    required = {'unique_id', 'ds', SCALABLE_MODEL_NAME}
    if not required.issubset(predictions.columns):
        raise NHITSInferenceError('NHITS_scalable produced an unexpected prediction schema')
    canonical_id = _canonical(series_id)
    selected = predictions.loc[predictions['unique_id'].astype(str) == canonical_id].sort_values('ds')
    if len(selected) != horizon or selected.duplicated(['unique_id', 'ds']).any():
        raise NHITSInferenceError(f'{canonical_id!r} did not produce exactly 28 unique predictions')
    expected_dates = [M5_FIRST_DATE + timedelta(days=origin_day + step - 1)
                      for step in range(1, horizon + 1)]
    actual_dates = [d.date() for d in __import__('pandas').to_datetime(selected['ds'])]
    if actual_dates != expected_dates:
        raise NHITSInferenceError('NHITS_scalable prediction dates do not align with the requested origin')
    raw = [float(value) for value in selected[SCALABLE_MODEL_NAME]]
    if any(not math.isfinite(value) for value in raw):
        raise NHITSInferenceError('NHITS_scalable returned nonfinite predictions')
    clipped = sum(value < 0 for value in raw)
    values = [max(0.0, value) for value in raw]
    metadata = _metadata(path, origin_day)
    return {'series_id': series_id, 'model': SCALABLE_MODEL_NAME, 'horizon': horizon,
            'forecast': values, 'total_expected_demand': float(sum(values)),
            'metadata': {'checkpoint_path': str(path), 'training_cutoff_day': origin_day,
                         'forecast_origin_day': origin_day,
                         'forecast_start_date': str(actual_dates[0]),
                         'forecast_end_date': str(actual_dates[-1]),
                         'series_id_canonical': canonical_id, 'inference_device': 'cpu',
                         'negative_policy': 'clip_to_zero_before_scoring_and_final_output',
                         'negative_predictions_clipped': clipped,
                         'history_source': 'caller-supplied origin-truncated M5 history',
                         'historical_observations': len(history),
                         'training_population': metadata['run_size'],
                         'train_series_count': metadata['train_series_count'],
                         'series_in_training_population': True,
                         'unseen_generalization_claim': False}}
