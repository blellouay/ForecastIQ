import numpy as np
import pandas as pd
import pytest
from src.evaluation.evaluation_panel import (PanelProtocol,create_evaluation_panel,validate_panel,
    prediction_records,assert_unseen_isolation,PREDICTION_COLUMNS)

def synthetic_sources(tmp_path):
    days={f"d_{i}":[i%3,(i+1)%3,(i+2)%3,(i+3)%3] for i in range(1,1914)}
    sales=pd.DataFrame({"item_id":["a","b","c","d"],"store_id":["s"]*4,**days})
    sales.insert(0,"series_id",sales.item_id+"__"+sales.store_id)
    meta=pd.DataFrame({"series_id":sales.series_id,"demand_type":["Smooth","Erratic","Intermittent","Lumpy"]})
    sp,mp=tmp_path/"sales.csv",tmp_path/"meta.csv";sales.to_csv(sp,index=False);meta.to_csv(mp,index=False)
    return sp,mp,sales

def test_panel_is_reproducible_and_valid(tmp_path):
    sp,mp,_=synthetic_sources(tmp_path);protocol=PanelProtocol(per_category=1)
    a=create_evaluation_panel(sp,mp,tmp_path/"a.parquet",protocol=protocol)
    b=create_evaluation_panel(sp,mp,tmp_path/"b.parquet",protocol=protocol)
    assert a.series_id.tolist()==b.series_id.tolist();assert validate_panel(a,protocol)

def test_prediction_schema_and_alignment(tmp_path):
    sp,mp,_=synthetic_sources(tmp_path);protocol=PanelProtocol(per_category=1)
    panel=create_evaluation_panel(sp,mp,tmp_path/"panel.parquet",protocol=protocol)
    records=prediction_records(panel,np.zeros((4,28)),"Naive","seen","x",protocol=protocol)
    assert records.columns.tolist()==PREDICTION_COLUMNS and len(records)==112
    assert records.groupby("series_id").horizon_step.nunique().eq(28).all()
    with pytest.raises(ValueError): prediction_records(panel,np.zeros((4,27)),"Naive","seen","x",protocol=protocol)
    bad=np.zeros((4,28));bad[0,0]=np.nan
    with pytest.raises(ValueError): prediction_records(panel,bad,"DeepAR","seen","x",protocol=protocol)

def test_unseen_isolation():
    assert assert_unseen_isolation(["a"],["b"])
    with pytest.raises(ValueError): assert_unseen_isolation(["a","b"],["b"])
