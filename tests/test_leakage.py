import numpy as np
import pandas as pd
from src.features.feature_engineering import fixed_origin_frame
from src.data.demand_classification import classify_demand

def test_future_changes_do_not_change_features_or_category():
    rng=np.random.default_rng(2); panel=rng.poisson(1,(4,120)).astype(float); origin=80
    before=fixed_origin_frame(panel,origin,28); labels=[classify_demand(x[:origin]) for x in panel]
    panel[:,origin:]+=999
    pd.testing.assert_frame_equal(before,fixed_origin_frame(panel,origin,28))
    assert labels==[classify_demand(x[:origin]) for x in panel]
