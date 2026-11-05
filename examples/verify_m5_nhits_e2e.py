"""Leakage-safe, real-M5 end-to-end verification of NHITS_scalable.

No model is trained. Selection uses targets ending at d_1885; d_1886--d_1913
is opened only after the agent has returned its selected final forecast.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.model_router import demand_profile
from src.agents.nhits_inference import (SCALABLE_ORIGINS, scalable_checkpoint_path,
                                        predict_nhits_scalable)
from src.agents.workflow import build_workflow
from src.features.m5_lightgbm import feature_provider

ORIGIN = 1885
HORIZON = 28
FOLD_ORIGINS = [1801, 1829, 1857]
TARGET_PATTERN_COUNTS = {'Smooth': 2, 'Erratic': 1, 'Intermittent': 1, 'Lumpy': 1}


def _mapped_ids(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding='utf-8'))
    mappings = [value for value in document.values() if isinstance(value, dict)
                and value and all(isinstance(code, int) and not isinstance(code, bool)
                                  for code in value.values())]
    if len(mappings) != 1:
        raise RuntimeError('Expected one frozen series-to-code mapping')
    return set(mappings[0])


def _canonical(raw_id: str) -> str:
    base = raw_id.removesuffix('_validation').removesuffix('_evaluation')
    parts = base.rsplit('_', 2)
    return parts[0] + '__' + parts[1] + '_' + parts[2]


def choose_series(sales_path: Path, mapping_path: Path) -> list[dict]:
    """Choose deterministic mapped series so ID-bound LightGBM is eligible."""
    mapped = _mapped_ids(mapping_path)
    selected: list[dict] = []
    counts = {name: 0 for name in TARGET_PATTERN_COUNTS}
    with sales_path.open('r', newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if _canonical(row['id']) not in mapped:
                continue
            history = [float(row[f'd_{day}']) for day in range(1, ORIGIN + 1)]
            profile = demand_profile(history)
            kind = profile['demand_type']
            if kind not in counts or counts[kind] >= TARGET_PATTERN_COUNTS[kind]:
                continue
            final_actual = [float(row[f'd_{day}'])
                            for day in range(ORIGIN + 1, ORIGIN + HORIZON + 1)]
            selected.append({'series_id': row['id'], 'demand_type': kind,
                             'adi': profile['adi'], 'cv2': profile['cv2'],
                             'history': history,
                             'final_actual': final_actual})
            counts[kind] += 1
            if counts == TARGET_PATTERN_COUNTS:
                break
    if counts != TARGET_PATTERN_COUNTS:
        raise RuntimeError(f'Could not find requested demand-pattern coverage: {counts}')
    return selected


def verify_checkpoints(root: Path) -> list[dict]:
    verified = []
    for origin in SCALABLE_ORIGINS:
        path = scalable_checkpoint_path(root, origin)
        metadata = json.loads((path / 'retsci_metadata.json').read_text(encoding='utf-8'))
        verified.append({'origin': origin, 'alias': metadata['alias'],
                         'training_cutoff': metadata['training_cutoff'],
                         'path': str(path)})
    return verified


def verify_lightgbm_cutoffs(*registry_paths: Path) -> list[dict]:
    evidence = []
    for registry_path in registry_paths:
        document = json.loads(registry_path.read_text(encoding='utf-8'))
        for model_name, model in document['models'].items():
            for origin_text, snapshot in model['snapshots'].items():
                origin = int(origin_text)
                cutoff = snapshot['trained_through_day']
                if snapshot['forecast_origin_day'] != origin or cutoff > origin:
                    raise RuntimeError(f'Future-trained snapshot: {model_name} at {origin}')
                evidence.append({'registry': str(registry_path), 'model': model_name,
                                 'origin': origin, 'trained_through_day': cutoff})
    return evidence


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sales_path = root / 'data/raw/sales_train_validation.csv'
    nhits_root = root / 'models/nhits_full_m5/full_all'
    registry_path = root / 'models/registry.json'
    mapping_path = root / 'models/lightgbm/series_mapping.json'
    output_dir = root / 'results/nhits_e2e'
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_evidence = verify_checkpoints(nhits_root)
    cutoff_evidence = verify_lightgbm_cutoffs(
        registry_path, root / 'models/generalizable_lightgbm/registry.json')
    candidates = choose_series(sales_path, mapping_path)
    graph = build_workflow(ml_registry_path=registry_path,
                           ml_feature_provider=feature_provider)
    details = []
    summary = []
    for index, candidate in enumerate(candidates, start=1):
        series_id = candidate['series_id']
        print(f'[{index}/5] {series_id} ({candidate["demand_type"]})', flush=True)
        result = graph.invoke({
            'series_id': series_id,
            'm5_sales_path': str(sales_path),
            'forecast_origin_day': ORIGIN,
            'horizon': HORIZON,
            'model': 'auto',
            'validation_folds': 3,
            'min_train_days': 90,
            'nhits_scalable_root': str(nhits_root),
        })
        if result.get('status') != 'completed':
            raise RuntimeError(f'{series_id} failed: {result.get("error")}')
        evidence = result['selection_evidence']
        validation_mae = evidence['validation_mae']
        if 'NHITS_scalable' not in validation_mae:
            raise RuntimeError(f'N-HiTS excluded for {series_id}: {evidence.get("nhits_excluded")}')
        if evidence.get('fold_origin_days') != FOLD_ORIGINS:
            raise RuntimeError(f'Unexpected fold origins: {evidence.get("fold_origin_days")}')
        if not all(name in validation_mae for name in
                   ('Average','Naive','SeasonalNaive','SES','Croston','SBA','TSB')):
            raise RuntimeError('Not all seven statistical models were compared')
        forecast = result['forecast_result']
        values = np.asarray(forecast['forecast'], dtype=float)
        if values.shape != (HORIZON,) or not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError('Selected final forecast contains invalid values')
        final_actual = np.asarray(candidate['final_actual'], dtype=float)
        final_test_mae = float(np.abs(values - final_actual).mean())
        nhits_final = predict_nhits_scalable(
            series_id, ORIGIN, candidate['history'], HORIZON, str(nhits_root))
        nhits_values = np.asarray(nhits_final['forecast'], dtype=float)
        nhits_meta = nhits_final['metadata']
        if (nhits_values.shape != (HORIZON,) or not np.isfinite(nhits_values).all()
                or (nhits_values < 0).any()
                or nhits_meta['series_id_canonical'] != _canonical(series_id)
                or nhits_meta['forecast_start_date'] != '2016-03-28'
                or nhits_meta['forecast_end_date'] != '2016-04-24'):
            raise RuntimeError(f'Invalid final N-HiTS diagnostic for {series_id}')
        selected = result['selected_model']
        selected_validation_mae = float(validation_mae[selected])
        row = {
            'series_id': series_id,
            'demand_type': candidate['demand_type'],
            'adi': candidate['adi'],
            'cv2': candidate['cv2'],
            'eligible_models': sorted(validation_mae),
            'validation_mae': validation_mae,
            'fold_mae': evidence.get('fold_mae'),
            'fold_origins': evidence['fold_origin_days'],
            'selected_model': selected,
            'selected_validation_mae': selected_validation_mae,
            'expected_28_day_demand': float(forecast['total_expected_demand']),
            'final_test_mae_d1886_d1913': final_test_mae,
            'final_actual_total': float(final_actual.sum()),
            'nhits_excluded': evidence.get('nhits_excluded'),
            'lightgbm_excluded': evidence.get('ml_excluded'),
            'generalizable_excluded': evidence.get('generalizable_excluded'),
            'selected_forecast_metadata': forecast.get('metadata'),
            'nhits_final_diagnostic': {
                'total_expected_demand': nhits_final['total_expected_demand'],
                'negative_predictions_clipped': nhits_meta['negative_predictions_clipped'],
                'forecast_start_date': nhits_meta['forecast_start_date'],
                'forecast_end_date': nhits_meta['forecast_end_date'],
                'series_id_canonical': nhits_meta['series_id_canonical'],
                'all_finite_and_nonnegative_after_policy': True,
            },
        }
        details.append(row)
        summary.append({key: row[key] for key in
                        ('series_id','demand_type','selected_model',
                         'selected_validation_mae','expected_28_day_demand',
                         'final_test_mae_d1886_d1913','final_actual_total')})
        print('  MAE:', ', '.join(f'{name}={score:.6f}'
                                  for name, score in sorted(validation_mae.items())))
        print(f'  selected={selected}; expected={row["expected_28_day_demand"]:.6f}; '
              f'final_test_mae={final_test_mae:.6f}', flush=True)

    payload = {'selection_boundary': {'validation_fold_origins': FOLD_ORIGINS,
                                      'final_forecast_origin': ORIGIN,
                                      'final_test_days': [1886, 1913],
                                      'final_test_used_for_selection': False},
               'checkpoint_evidence': checkpoint_evidence,
               'lightgbm_cutoff_evidence': cutoff_evidence,
               'results': details}
    (output_dir / 'details.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    pd.DataFrame(summary).to_csv(output_dir / 'summary.csv', index=False)
    print('\nSUMMARY')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Wrote {output_dir / "details.json"}')
    print(f'Wrote {output_dir / "summary.csv"}')


if __name__ == '__main__':
    main()
