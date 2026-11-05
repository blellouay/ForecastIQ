import numpy as np
import pytest
from src.evaluation.metrics import *

def test_metrics_formulas():
    a=np.array([1,2,3]);p=np.array([2,2,1]);train=np.array([1,2,3,4])
    assert mae(a,p)==1 and mean_signed_bias(a,p)==pytest.approx(-1/3)
    assert wape(a,p)==pytest.approx(.5) and mase(a,p,train)==1
    assert underforecast_quantity(a,p)==2 and overforecast_quantity(a,p)==1
    assert inventory_cost_proxy(a,p,5,1)==11
def test_undefined_scaled_metrics_are_nan():
    assert np.isnan(mase([1],[1],[2,2])) and np.isnan(wape([0],[1]))
