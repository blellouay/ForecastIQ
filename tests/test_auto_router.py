import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.model_router import demand_profile, validation_cutoffs, select_by_rolling_validation
from src.agents.workflow import select_model, analyze_history
from src.agent_tools import ToolError


def fake_forecast(series_id, sales_history, model, horizon, alpha):
    value = {'TSB': 0., 'SES': 1., 'SBA': 2.}.get(model, 5.)
    return {'forecast': [value]*horizon}


def test_cv_cutoffs_are_history_only():
    assert validation_cutoffs(1885, 28, 3, 90) == [1801,1829,1857]
    assert validation_cutoffs(100, 28, 3, 90) == []


def test_model_uses_aggregated_fold_error():
    a = np.zeros(1885)
    with patch('src.agents.model_router.forecast_statistical', side_effect=fake_forecast):
        result = select_by_rolling_validation('A', a, candidates=['TSB', 'SES'], n_folds=3)
    assert result['model'] == 'TSB'
    assert result['fold_cutoffs'] == [1801,1829,1857]
    assert len(result['fold_mae']['TSB']) == 3


def test_future_data_cannot_change_selection():
    a = np.zeros(1885)
    future = np.full(28, 999.)
    # Outer loader never supplies future actuals; input is identical.
    with patch('src.agents.model_router.forecast_statistical', side_effect=fake_forecast):
        a_result = select_by_rolling_validation('A', a, candidates=['TSB','SES'])
        b_result = select_by_rolling_validation('A', np.concatenate([a, future])[:1885], candidates=['TSB','SES'])
    assert a_result == b_result


def test_short_history_fallback_is_explicit():
    result = select_by_rolling_validation('A', [0, 1, 0], n_folds=3)
    assert result['model'] == 'TSB' and result['method'] == 'fallback'


def test_demand_classification():
    a = demand_profile([0, 0, 1, 1, 1, 1])
    assert a['demand_type'] == 'Smooth'
    assert demand_profile([0,0,0])['demand_type'] == 'Insufficient history'
    assert analyze_history({'series_id':'A','sales_history':[0,0,1,1,1]})['diagnostics']['demand_type']=='Smooth'


def test_auto_select_node():
    with patch('src.agents.model_router.forecast_statistical', side_effect=fake_forecast):
        out = select_model({'series_id':'A','sales_history':[0.]*1885,'model':'auto'})
    assert out['selected_model']=='TSB' and out['status']=='selected'
    assert out['selection_evidence']['method']=='historical_rolling_mae'


def test_invalid_candidates_and_params():
    with pytest.raises(ToolError):
        select_by_rolling_validation('A', [1]*200, candidates=['Unknown'])
    with pytest.raises(ToolError):
        validation_cutoffs(500,28,n_folds=0)
