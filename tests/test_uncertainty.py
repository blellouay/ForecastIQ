import pandas as pd
import pytest

from src.evaluation.uncertainty import paired_hierarchical_comparison


def _records(origins=(10,20)):
    rows=[]
    for origin in origins:
        for series,item in (("s1","i1"),("s2","i1"),("s3","i2")):
            for day in (1,2):
                for model,pred in (("a",0.0),("b",1.0)):
                    rows.append({"scenario":"seen","series_id":series,"item_id":item,
                                 "forecast_origin":origin,"target_day":origin+day,
                                 "model":model,"actual":0.0,"prediction":pred})
    return pd.DataFrame(rows)


def test_hierarchical_comparison_reports_population_and_is_reproducible():
    a=paired_hierarchical_comparison(_records(),"a","b",n_bootstrap=100,seed=7)
    b=paired_hierarchical_comparison(_records(),"a","b",n_bootstrap=100,seed=7)
    assert a==b
    assert a["matched_series"]==3 and a["matched_products"]==2
    assert a["matched_origins"]==2 and a["matched_observations"]==12
    assert a["mean_difference"]==-1.0


def test_hierarchical_comparison_rejects_one_origin():
    with pytest.raises(ValueError,match="two forecast origins"):
        paired_hierarchical_comparison(_records((10,)),"a","b")
