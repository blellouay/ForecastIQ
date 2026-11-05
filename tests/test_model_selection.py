import pandas as pd
from src.evaluation.model_selection import select_models
def test_strategies_and_deterministic_tie():
    df=pd.DataFrame({"series_id":["s1","s1","s2","s2"],"demand_type":["x"]*4,
                     "model":["B","A","B","A"],"validation_loss":[1,1,2,3]})
    assert select_models(df,"global")["__global__"]=="B"
    assert select_models(df,"per_series")["s1"]=="A"

def test_incomplete_validation_origins_are_excluded():
    df=pd.DataFrame({"series_id":["s1"]*5,"demand_type":["x"]*5,
        "model":["partial","partial","complete","complete","complete"],
        "validation_origin":[10,20,10,20,30],"validation_loss":[0,0,1,1,1]})
    assert select_models(df,"global",expected_origins=(10,20,30))["__global__"]=="complete"
