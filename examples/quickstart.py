"""Fast, offline ForecastIQ demonstration using synthetic sales history."""
from src.agents.workflow import build_workflow


def main() -> None:
    history = [0, 1, 0, 2, 0, 0, 1] * 28
    result = build_workflow(ml_registry_path=None, generalizable_registry_path=None).invoke({
        'series_id': 'DEMO_SERIES', 'sales_history': history, 'model': 'auto',
        'horizon': 28, 'validation_folds': 3, 'min_train_days': 90,
        'nhits_scalable_root': 'models/not-required-for-quickstart',
    })
    if result.get('status') != 'completed':
        raise RuntimeError(result.get('error', 'ForecastIQ quick start failed'))
    print('ForecastIQ offline quick start (synthetic demonstration)')
    print('Validation MAE:')
    for model, mae in sorted(result['selection_evidence']['validation_mae'].items()):
        print(f'  {model}: {mae:.6f}')
    print('Selected model:', result['selected_model'])
    print('Expected 28-day demand:', f"{result['forecast_result']['total_expected_demand']:.3f}")
    print('This synthetic result is not an M5 benchmark.')


if __name__ == '__main__':
    main()
