import pandas as pd
import pytest
from src.evaluation.comparison import assert_matched_predictions
def records(model,actual=(1,2)):
    return pd.DataFrame({"scenario":["seen"]*2,"model":[model]*2,"series_id":["a"]*2,"forecast_origin":[10]*2,
      "target_day":[11,12],"horizon_step":[1,2],"actual":actual,"prediction":[1,1]})
def test_matched_and_mismatched():
    assert assert_matched_predictions(pd.concat([records("a"),records("b")]))
    with pytest.raises(ValueError): assert_matched_predictions(pd.concat([records("a"),records("b",(1,3))]))
