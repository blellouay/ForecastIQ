"""Run from project root: python -m examples.run_step2"""
import json
from src.agents import build_workflow

# Explicitly synthetic demonstration data. NOT an M5 benchmark.
sales = [0, 0, 2, 0, 0, 1, 0] * 20
result = build_workflow().invoke({
    'series_id': 'DEMO_ITEM__DEMO_STORE',
    'sales_history': sales,
    'model': 'TSB',
    'horizon': 28,
    'alpha': 0.2,
})
if result.get('status') == 'error':
    raise RuntimeError(result['error'])
print(json.dumps({
    'status': result['status'],
    'diagnostics': result['diagnostics'],
    'model': result['selected_model'],
    'forecast': result['forecast_result']['forecast'],
    'total': result['forecast_result']['total_expected_demand'],
    'explanation': result['explanation'],
}, indent=2))
