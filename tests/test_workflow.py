import csv
import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.workflow import (load_history, analyze_history, select_model,
                                 generate_forecast, review_forecast, _read_m5_history)


def test_inline_load():
    s = load_history({'series_id':'A','sales_history':[0, 2, 0]})
    assert s['status'] == 'history_loaded'


def test_source_exclusivity():
    assert load_history({'series_id':'A','sales_history':[1], 'm5_sales_path':'x'})['status'] == 'error'
    assert load_history({'series_id':'A'})['status'] == 'error'


def test_default_policy_not_claimed_best():
    s = select_model({})
    assert s['selected_model'] == 'TSB' and 'NOT selected' in s['selection_reason']


def test_explicit_model():
    s = select_model({'model':'SES'})
    assert s['selected_model'] == 'SES'
    assert select_model({'model':'NHiTS'})['status'] == 'error'


def test_analysis():
    a = analyze_history({'series_id':'A','sales_history':[0,2,0,4]})
    assert a['diagnostics']['zero_ratio'] == 0.5


def test_forecast_adapter_mock():
    def fake(**kwargs):
        return {'series_id':kwargs['series_id'],'model':kwargs['model'], 'horizon':28,
                'forecast':[2.0]*28,'total_expected_demand':56.0}
    state = {'series_id':'A','selected_model':'TSB','sales_history':[0,2,0]}
    with patch('src.agents.workflow.forecast_statistical', side_effect=fake):
        result = generate_forecast(state)
    assert result['status'] == 'forecasted'
    reviewed = review_forecast({**state, **result,'selection_reason':'Test policy'})
    assert reviewed['status'] == 'completed'


def test_bad_forecast_rejected():
    state = {'series_id':'A','selected_model':'TSB','selection_reason':'demo',
             'forecast_result':{'series_id':'A','model':'TSB','horizon':28,
                                'forecast':[math.nan]*28,'total_expected_demand':0}}
    assert review_forecast(state)['status'] == 'error'


def test_m5_cutoff_no_future(tmp_path):
    f = tmp_path/'sales.csv'
    with f.open('w',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=['id','d_1','d_2','d_3'])
        writer.writeheader(); writer.writerow({'id':'A','d_1':0,'d_2':2,'d_3':999})
    assert _read_m5_history(str(f),'A',2) == [0.,2.]
    assert load_history({'series_id':'A','m5_sales_path':str(f),'forecast_origin_day':2})['sales_history']==[0.,2.]
    assert load_history({'series_id':'A','m5_sales_path':str(f)})['status']=='error'


def test_graph_integration_when_installed():
    import pytest
    pytest.importorskip('langgraph')
    from src.agents import build_workflow
    def fake(**kwargs):
        return {'series_id':kwargs['series_id'],'model':kwargs['model'], 'horizon':28,
                'forecast':[1.]*28,'total_expected_demand':28.}
    with patch('src.agents.workflow.forecast_statistical',side_effect=fake):
        result = build_workflow().invoke({'series_id':'A','sales_history':[0,1,0]})
    assert result['status']=='completed'
    assert len(result['forecast_result']['forecast'])==28
    assert build_workflow().invoke({'series_id':'A'})['status']=='error'
