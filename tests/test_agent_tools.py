import sys,types
import numpy as np
import pytest
from src.agent_tools import describe_history, forecast_statistical, ToolError, available_models


def test_history():
    data=describe_history('A__CA_1',[0,2,0,4])
    assert data['zero_ratio']==0.5 and data['mean_positive_quantity']==3

@pytest.mark.parametrize('bad', [[],[-1], [float('nan')],[[1,2]], [float('inf')]])
def test_invalid_history(bad):
    with pytest.raises(ToolError): describe_history('x',bad)

@pytest.mark.parametrize('h', [0,366,1.5,True])
def test_invalid_horizon(h):
    with pytest.raises(ToolError): forecast_statistical('x',[1,0],horizon=h)


def test_statistical_adapter(monkeypatch):
    fake=types.ModuleType('src.forecasting.statistical_models')
    calls=[]
    def forecast_matrix(model, hist, horizon, **kwargs):
        calls.append((model,hist.shape,horizon,kwargs))
        return np.ones((1,horizon))*2
    fake.forecast_matrix=forecast_matrix
    monkeypatch.setitem(sys.modules,'src.forecasting',types.ModuleType('src.forecasting'))
    monkeypatch.setitem(sys.modules,'src.forecasting.statistical_models',fake)
    result=forecast_statistical('FOODS_1_001__CA_1',[0,1,2],'TSB',28,alpha=.15)
    assert len(result['forecast'])==28 and result['total_expected_demand']==56
    assert calls==[('TSB',(1,3),28,{'alpha':.15})]
    assert 'TSB' in available_models()['statistical']


def test_bad_model_output(monkeypatch):
    fake=types.ModuleType('src.forecasting.statistical_models')
    fake.forecast_matrix=lambda *a,**k: np.array([float('nan')]*28)
    monkeypatch.setitem(sys.modules,'src.forecasting',types.ModuleType('src.forecasting'))
    monkeypatch.setitem(sys.modules,'src.forecasting.statistical_models',fake)
    with pytest.raises(ToolError): forecast_statistical('x',[0,1], 'TSB')
