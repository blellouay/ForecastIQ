"""Run from project root: python -m examples.run_step3"""
import json
from src.agents import build_workflow

# A small synthetic series for smoke testing; replace with M5 CSV in your script.
history = ([0, 0, 2, 0, 1, 0, 0] * 35)[:245]
result = build_workflow().invoke({
    'series_id': 'DEMO_ITEM__DEMO_STORE',
    'sales_history': history,
    'model': 'auto',
    'horizon': 28,
    'validation_folds': 3,
    'min_train_days': 90,
})
summary = {
    'status': result.get('status'),
    'demand_type': result.get('diagnostics', {}).get('demand_type'),
    'selected_model': result.get('selected_model'),
    'selection_evidence': result.get('selection_evidence'),
    'forecast_total': result.get('forecast_result', {}).get('total_expected_demand'),
    'explanation': result.get('explanation'),
    'error': result.get('error') or None,
}
print(json.dumps(summary, indent=2))
