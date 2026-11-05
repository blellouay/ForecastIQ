import numpy as np
import pytest
from src.forecasting.statistical_models import *

CASES=[np.ones(20)*3,np.zeros(20),np.r_[np.zeros(30),2,0,0,4],np.array([5.]),
       np.array([0,0,2,0,0,0,3,0,0,1]),np.array([1,20,1,30,0,0,2])]

@pytest.mark.parametrize("model",list(MODELS))
@pytest.mark.parametrize("history",CASES)
@pytest.mark.parametrize("horizon",[1,7,28])
def test_forecasts_are_finite_nonnegative(model,history,horizon):
    kwargs={"alpha":.2} if model in {"SES","Croston","SBA","TSB"} else {}
    result=forecast(model,history,horizon,**kwargs)
    assert result.shape==(horizon,) and np.isfinite(result).all() and (result>=0).all()

def test_constant_demand_reference_behavior():
    y=np.full(50,4.)
    for model in ["Average","Naive","SES","Croston"]: assert np.allclose(forecast(model,y,5,**({"alpha":.2} if model in {"SES","Croston"} else {})),4)

@pytest.mark.parametrize("alpha",[.1,.3,1.0])
def test_scalar_matrix_equivalence(alpha):
    matrix=np.array([[0,0,0,0,0,0,0,0], [0,0,2,0,0,4,0,0], [5,0,0,0,0,0,0,0]],dtype=float)
    for model in MODELS:
        kwargs={"alpha":alpha} if model in {"SES","Croston","SBA","TSB"} else {}
        expected=np.vstack([forecast(model,row,6,**kwargs) for row in matrix])
        assert np.allclose(forecast_matrix(model,matrix,6,**kwargs),expected)

def test_invalid_inputs_rejected():
    with pytest.raises(ValueError): forecast_average([1,-1],2)
    with pytest.raises(ValueError): forecast_ses([1,2],2,alpha=0)
