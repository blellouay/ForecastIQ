from pathlib import Path
import pytest
from src.forecasting.cold_start import detect_forecasting_scenario,comparable_product_forecast
from src.agents.workflow import build_workflow

ROOT=Path(__file__).resolve().parents[1]

def test_scenario_detection():
    assert detect_forecasting_scenario('FOODS_1_999_CA_1_validation',[])['scenario']=='no_history_cold_start'
    assert detect_forecasting_scenario('FOODS_1_999_CA_1_validation',[0]*20)['scenario']=='short_history'
    assert detect_forecasting_scenario('FOODS_1_999_CA_1_validation',[0]*56)['scenario']=='unseen_series_with_history'
    assert detect_forecasting_scenario('FOODS_2_128_TX_3_validation',[0]*56)['scenario']=='known_series'

def test_cold_start_requires_metadata():
    with pytest.raises(ValueError,match='metadata requires'):
        comparable_product_forecast('NEW',{},ROOT/'data/processed/m5_sample_400.csv',1885)

def test_comparable_baseline_uses_historical_cutoff():
    result=comparable_product_forecast('FOODS_1_999_CA_1_validation',
        {'category':'FOODS','department':'FOODS_1','store':'CA_1'},
        ROOT/'data/processed/m5_sample_400.csv',1885)
    assert len(result['forecast'])==28
    assert result['metadata']['history_cutoff_day']==1885
    assert result['metadata']['broader_category_products']>0
    assert result['total_expected_demand']>=0

def test_langgraph_routes_no_history_to_cold_start():
    graph=build_workflow()
    result=graph.invoke({'series_id':'FOODS_1_999_CA_1_validation','sales_history':[],
        'forecast_origin_day':1885,'horizon':28,'model':'auto',
        'product_metadata':{'category':'FOODS','department':'FOODS_1','store':'CA_1'},
        'comparable_sales_path':str(ROOT/'data/processed/m5_sample_400.csv')})
    assert result['status']=='completed'
    assert result['forecasting_scenario']['scenario']=='no_history_cold_start'
    assert result['selected_model']=='ColdStartComparable'
    assert result['forecast_result']['metadata']['scenario']=='simulated_no_history'
