"""Truthful inference and automatic-selection eligibility reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.model_router import validation_cutoffs
from src.agents.nhits_inference import TRAINING_CUTOFF_DAY
from src.agents.nhits_inference import (SCALABLE_MODEL_NAME, scalable_checkpoint_path,
                                        NHITSInferenceError)
from src.agent_tools.forecasting_tools import ToolError

STATISTICAL = ('Average','Naive','SeasonalNaive','SES','Croston','SBA','TSB')
LIGHTGBM = ('DirectLightGBM','HurdleLightGBM')


def assess_model_availability(series_id: str, history: list[float], origin: int,
                              horizon: int = 28, registry_path=None,
                              feature_provider=None,
                              nhits_checkpoint_path='models/neuralforecast/nhits_small',
                              generalizable_registry_path=None,
                              nhits_scalable_root='models/nhits_full_m5/full_all') -> list[dict[str, Any]]:
    folds = validation_cutoffs(len(history), horizon, 3, 90)
    statistical_ok = len(folds) == 3
    has_history=len(history)>0
    rows = [
        {'model': name, 'family': 'statistical', 'inference_eligible': has_history,
         'selection_eligible': statistical_ok,
         'reason': 'Matched rolling-origin validation is available.' if statistical_ok else
                   'Insufficient history for three matched validation folds.'}
        for name in STATISTICAL
    ]
    absolute_folds = [origin - (len(history) - cutoff) for cutoff in folds]
    for name in LIGHTGBM:
        inference = selection = False
        reason = 'No trusted registry and feature provider supplied.'
        if registry_path and feature_provider:
            try:
                from src.agents.ml_routing import _artifact, _registry, _snapshot
                document, base = _registry(registry_path)
                final = _snapshot(document, name, origin)
                feature_provider(series_id, history, origin, horizon, final['features'])
                paths = ([final.get('model_path')] if final['kind'] == 'direct' else
                         [final.get('classifier_path'), final.get('regressor_path')])
                for path in paths:
                    _artifact(base, path)
                inference = True
                for fold_origin in absolute_folds:
                    _snapshot(document, name, fold_origin)
                selection = len(absolute_folds) == 3
                reason = ('Exact-origin artifacts and matched historical folds are available.'
                          if selection else 'Inference works, but three matched folds are unavailable.')
            except (ToolError, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
        rows.append({'model':name, 'family':'machine_learning',
                     'inference_eligible':inference, 'selection_eligible':selection,
                     'reason':reason})
    from src.agents.generalizable_routing import NAMES as GENERAL_NAMES, predict_generalizable_snapshot
    for name in GENERAL_NAMES:
        eligible=False;reason='No generalizable LightGBM registry supplied.'
        if generalizable_registry_path:
            try:
                predict_generalizable_snapshot(series_id,history,origin,horizon,name,
                                                generalizable_registry_path)
                for fold_origin,cutoff in zip(absolute_folds,folds):
                    predict_generalizable_snapshot(series_id,history[:cutoff],fold_origin,horizon,name,
                                                    generalizable_registry_path)
                eligible=len(folds)==3
                reason='ID-free pilot artifacts and three matched historical snapshots are available.'
            except (ToolError,ValueError,TypeError,KeyError,OSError) as exc:reason=str(exc)
        rows.append({'model':name,'family':'machine_learning','inference_eligible':eligible,
                     'selection_eligible':eligible,'reason':reason})
    root = Path(nhits_checkpoint_path)
    needed = ('configuration.json','configuration.safetensors','dataset.json',
              'dataset.safetensors','NHITS_small_0.safetensors')
    complete = all((root / name).is_file() for name in needed)
    nhits_inference = complete and horizon == 28 and origin >= TRAINING_CUTOFF_DAY and len(history)>=56
    if not complete:
        nhits_reason = 'Saved N-HiTS inference bundle is incomplete.'
    elif horizon != 28:
        nhits_reason = 'Saved N-HiTS supports only a 28-day horizon.'
    elif origin < TRAINING_CUTOFF_DAY:
        nhits_reason = f'Weights trained through d_{TRAINING_CUTOFF_DAY} are invalid at this earlier origin.'
    elif len(history)<56:
        nhits_reason='N-HiTS requires sufficient supplied history; cold-start inference is unsupported.'
    else:
        nhits_reason = ('Supplied-history CPU inference is available; automatic ranking is excluded '
                        'because matched pre-origin checkpoint snapshots do not exist.')
    rows.append({'model':'NHITS_small','family':'deep_learning',
                 'inference_eligible':nhits_inference, 'selection_eligible':False,
                 'reason':nhits_reason})
    scalable_inference = scalable_selection = False
    try:
        scalable_checkpoint_path(nhits_scalable_root, origin)
        scalable_inference = horizon == 28 and len(history) == origin and len(history) >= 56
        if not scalable_inference:
            raise NHITSInferenceError('NHITS_scalable requires horizon 28 and complete d_1-through-origin history')
        for fold_origin in absolute_folds:
            scalable_checkpoint_path(nhits_scalable_root, fold_origin)
        scalable_selection = len(absolute_folds) == 3
        scalable_reason = ('Exact-origin full-panel checkpoint is available for inference and all three '
                           'matched historical validation snapshots are available.' if scalable_selection else
                           'Exact-origin inference is available, but three matched folds are unavailable.')
    except (NHITSInferenceError, OSError, ValueError, TypeError) as exc:
        scalable_reason = str(exc)
    rows.append({'model': SCALABLE_MODEL_NAME, 'family': 'deep_learning',
                 'inference_eligible': scalable_inference,
                 'selection_eligible': scalable_selection,
                 'reason': scalable_reason})
    rows.append({'model':'DeepAR','family':'deep_learning',
                 'inference_eligible':False, 'selection_eligible':False,
                 'reason':'Training checkpoints exist, but no verified NeuralForecast inference bundle '
                          'with dataset identity and cutoff provenance is exported.'})
    rows.append({'model':'ColdStartComparable','family':'cold_start',
                 'inference_eligible':not has_history,'selection_eligible':False,
                 'reason':'Available only for no-history requests with explicit category, department, store, and a comparable-sales dataset.'})
    return rows
