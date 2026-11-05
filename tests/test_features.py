import numpy as np
from src.features.feature_engineering import historical_snapshot,fixed_origin_frame

def test_snapshot_values():
    y=np.arange(1,61,dtype=float)[None,:]
    f=historical_snapshot(y)
    assert f["lag_1"][0]==60 and f["lag_7"][0]==54 and f["mean_7"][0]==57
def test_frame_shape():
    y=np.ones((3,100)); assert fixed_origin_frame(y,70,28).shape[0]==84
