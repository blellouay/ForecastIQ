"""Demonstrate unseen-with-history and no-history hybrid routing."""
from pathlib import Path
from src.agents.workflow import build_workflow
from src.features.m5_lightgbm import feature_provider

ROOT=Path(__file__).resolve().parents[1]

def show(label,result):
    print('\n'+label)
    print('Scenario:',result.get('forecasting_scenario'))
    print('Selected:',result.get('selected_model'))
    print('Eligible MAE:',(result.get('selection_evidence') or {}).get('validation_mae',{}))
    print('Excluded:',{**((result.get('selection_evidence') or {}).get('ml_excluded',{})),
                       **((result.get('selection_evidence') or {}).get('generalizable_excluded',{}))})
    print('Total:',(result.get('forecast_result') or {}).get('total_expected_demand'))
    print('Metadata:',(result.get('forecast_result') or {}).get('metadata'))

def main():
    graph=build_workflow(ml_feature_provider=feature_provider,ml_registry_path=ROOT/'models/registry.json')
    unseen=graph.invoke({'series_id':'FOODS_1_001_CA_1_validation',
        'm5_sales_path':str(ROOT/'data/raw/sales_train_validation.csv'),
        'forecast_origin_day':1885,'horizon':28,'model':'auto'})
    cold=graph.invoke({'series_id':'FOODS_1_999_CA_1_validation','sales_history':[],
        'forecast_origin_day':1885,'horizon':28,'model':'auto',
        'product_metadata':{'category':'FOODS','department':'FOODS_1','store':'CA_1'},
        'comparable_sales_path':str(ROOT/'data/processed/m5_sample_400.csv')})
    show('UNSEEN SERIES WITH HISTORY',unseen);show('SIMULATED NO-HISTORY PRODUCT',cold)

if __name__=='__main__':main()
