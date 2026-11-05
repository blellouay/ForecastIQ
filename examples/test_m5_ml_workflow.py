"""Real-data Step 4 demonstration with saved LightGBM and safe fallback."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.workflow import build_workflow
from src.features.m5_lightgbm import feature_provider


def main():
    graph = build_workflow(
        ml_registry_path=ROOT / "models/registry.json",
        ml_feature_provider=feature_provider,
    )
    result = graph.invoke({
        "series_id": "FOODS_1_001_CA_1_validation",
        "m5_sales_path": str(ROOT / "data/raw/sales_train_validation.csv"),
        "forecast_origin_day": 1885,
        "horizon": 28,
        "model": "auto",
        "validation_folds": 3,
        "min_train_days": 90,
    })
    evidence = result.get("selection_evidence", {})
    diagnostics = result.get("diagnostics", {})
    forecast = result.get("forecast_result", {})
    print("Demand category:", diagnostics.get("demand_type"))
    print("Available models:", sorted(evidence.get("validation_mae", {})))
    print("ML exclusion reasons:", evidence.get("ml_excluded", {}))
    print("Historical validation MAE:")
    for name, score in sorted(evidence.get("validation_mae", {}).items()):
        print(f"  {name}: {score:.6f}")
    print("Automatically selected model:", result.get("selected_model"))
    print("Forecast:", forecast.get("forecast"))
    print("Total expected demand:", forecast.get("total_expected_demand"))
    print("Explanation:", result.get("explanation") or result.get("error"))


if __name__ == "__main__":
    main()
