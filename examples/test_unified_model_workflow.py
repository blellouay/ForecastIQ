"""Real M5 unified eligibility and automatic-selection demonstration."""
from pathlib import Path

from src.agents.conversational import build_conversational_workflow
from src.features.m5_lightgbm import feature_provider


def main():
    root = Path(__file__).resolve().parents[1]
    graph = build_conversational_workflow(
        ml_feature_provider=feature_provider,
        llm_provider='offline',
    )
    conversation = graph.invoke({
        'request':'Forecast FOODS_1_001 in store CA_1 for 28 days using auto',
        'forecast_origin_day':1885,
        'm5_sales_path':str(root / 'data/raw/sales_train_validation.csv'),
        'ml_registry_path':str(root / 'models/registry.json'),
        'nhits_checkpoint_path':str(root / 'models/neuralforecast/nhits_small'),
    })
    result = conversation.get('forecast_result_state') or {}
    evidence = result.get('selection_evidence') or {}
    availability = result.get('model_availability') or []
    print('Requested product:', result.get('series_id'))
    print('Forecast origin:', 1885)
    print('Inference eligible:', [r['model'] for r in availability if r['inference_eligible']])
    print('Auto-selection eligible:', [r['model'] for r in availability if r['selection_eligible']])
    print('Excluded:', {r['model']:r['reason'] for r in availability if not r['selection_eligible']})
    print('Validation MAE:', evidence.get('validation_mae', {}))
    print('Selected model:', result.get('selected_model'))
    payload = result.get('forecast_result') or {}
    print('28-day forecast:', payload.get('forecast'))
    print('Expected total demand:', payload.get('total_expected_demand'))
    print('AI explanation:', conversation.get('ai_explanation') or conversation.get('llm_error'))


if __name__ == '__main__':
    main()
