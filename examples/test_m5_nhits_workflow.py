"""Run from project root: python -m examples.test_m5_nhits_workflow"""
from src.agents import build_workflow

graph = build_workflow()
result = graph.invoke({
    'series_id': 'FOODS_1_008_TX_1_validation',
    'm5_sales_path': 'data/raw/sales_train_validation.csv',  # Adjust if needed.
    'forecast_origin_day': 1857,  # Saved snapshot starts forecasting 2016-02-29.
    'horizon': 28,
    'model': 'NHITS_small',  # Explicit only; NOT part of auto model routing.
    'nhits_checkpoint_path': 'models/neuralforecast/nhits_small',
})
print('Status:',result.get('status'))
print('Explanation:',result.get('explanation') or result.get('error'))
if result.get('status')=='completed':
    forecast=result['forecast_result']
    print('Total:',forecast['total_expected_demand'])
    print('28-day forecast:',forecast['forecast'])
    print('Checkpoint metadata:',forecast['metadata'])
