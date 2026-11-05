import numpy as np
import pytest
from src.data.preparation import *

def test_active_history_helpers():
    y=np.array([0,0,2,0,1])
    assert first_positive_index(y)==2 and active_history_length(y)==3
    assert np.array_equal(apply_leading_zero_policy(y,"active"),[2,0,1])
    assert zero_sales_ratio(y,"active")==pytest.approx(1/3)
def test_all_zero_and_validation():
    assert first_positive_index([0,0]) is None and active_history_length([0,0])==0
    with pytest.raises(ValueError): validate_daily_sales([1,np.nan])
