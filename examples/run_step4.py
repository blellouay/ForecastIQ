"""Step 4: backwards-compatible auto selection with explicit saved-ML eligibility.
Run from project root: python -m examples.run_step4
"""
from src.agents import build_workflow

# Synthetic intermittent history only for demo. Real M5 invocation is in README.
history = ([0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0] * 35)

graph = build_workflow()  # no artifact registry supplied => safe statistical-only auto selection
result = graph.invoke({
    "series_id": "DEMO_ITEM__DEMO_STORE",
    "sales_history": history,
    "model": "auto",
    "horizon": 28,
    "validation_folds": 3,
    "min_train_days": 90,
})
print('Status:', result.get('status'))
print('Chosen:', result.get('selected_model'))
print('Exclusion reasons:', result.get('selection_evidence', {}).get('ml_excluded'))
print('Explanation:', result.get('explanation') or result.get('error'))
if result.get('status') == 'completed':
    print('28-day forecast:', result['forecast_result']['forecast'])
