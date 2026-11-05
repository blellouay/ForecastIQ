from src.agents import build_workflow

graph = build_workflow()

result = graph.invoke({
    "series_id": "FOODS_1_001_CA_1_validation",
    "m5_sales_path": "data/raw/sales_train_validation.csv",
    "forecast_origin_day": 1885,
    "horizon": 28,
    "model": "auto",
    "validation_folds": 3,
    "min_train_days": 90
})

print("Status:", result.get("status"))
print("Selected model:", result.get("selected_model"))
print("Explanation:", result.get("explanation") or result.get("error"))
print("ML availability:", result.get("selection_evidence", {}).get("ml_excluded"))

if result.get("status") == "completed":
    print("Forecast:", result["forecast_result"]["forecast"])