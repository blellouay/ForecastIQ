"""Real per-series comparison for the full-panel N-HiTS snapshots."""
from pathlib import Path

from src.agents.workflow import build_workflow
from src.features.m5_lightgbm import feature_provider


def main():
    root = Path(__file__).resolve().parents[1]
    graph = build_workflow(
        ml_registry_path=root / 'models/registry.json',
        ml_feature_provider=feature_provider,
    )
    result = graph.invoke({
        'series_id': 'FOODS_1_008_TX_1_validation',
        'm5_sales_path': str(root / 'data/raw/sales_train_validation.csv'),
        'forecast_origin_day': 1885,
        'horizon': 28,
        'model': 'auto',
        'validation_folds': 3,
        'min_train_days': 90,
        'nhits_scalable_root': str(root / 'models/nhits_full_m5/full_all'),
    })
    evidence = result.get('selection_evidence') or {}
    print('Fold origins:', evidence.get('fold_origin_days'))
    print('Validation MAE:')
    for name, value in sorted((evidence.get('validation_mae') or {}).items()):
        print(f'  {name}: {value:.6f}')
    print('N-HiTS fold MAE:', (evidence.get('fold_mae') or {}).get('NHITS_scalable'))
    print('N-HiTS exclusion:', (evidence.get('nhits_excluded') or {}).get('NHITS_scalable'))
    print('LightGBM exclusions:', evidence.get('ml_excluded'))
    print('Selected model:', result.get('selected_model'))
    forecast = result.get('forecast_result') or {}
    print('Final model:', forecast.get('model'))
    print('Final total:', forecast.get('total_expected_demand'))
    print('Final metadata:', forecast.get('metadata'))
    print('Error:', result.get('error'))


if __name__ == '__main__':
    main()
