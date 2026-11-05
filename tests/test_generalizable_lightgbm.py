import json
from pathlib import Path
import numpy as np,pandas as pd
import pytest
from src.features.generalizable_features import FEATURE_NAMES,feature_frame
from src.agents.generalizable_routing import predict_generalizable_snapshot

ROOT=Path(__file__).resolve().parents[1]

def test_features_have_no_series_code_and_no_future_leakage():
    dates=pd.date_range('2011-01-29',periods=200,freq='D');history=np.arange(112,dtype=float)%5
    first=feature_frame(['FOODS_1_001__CA_1'],[history],112,28,dates)
    second=feature_frame(['FOODS_1_001__CA_1'],[(np.r_[history,[999]*28])[:112]],112,28,dates)
    pd.testing.assert_frame_equal(first,second)
    assert list(first.columns)==FEATURE_NAMES
    assert 'series_code' not in first

@pytest.mark.parametrize('model',['GeneralizableDirectLightGBM','GeneralizableHurdleLightGBM'])
@pytest.mark.integration
def test_unseen_series_real_artifact_inference(model):
    raw=pd.read_csv(ROOT/'data/raw/sales_train_validation.csv')
    row=raw.loc[raw.id=='FOODS_1_001_CA_1_validation'].iloc[0]
    history=[float(row[f'd_{i}']) for i in range(1,1886)]
    result=predict_generalizable_snapshot('FOODS_1_001_CA_1_validation',history,1885,28,
        model,ROOT/'models/generalizable_lightgbm/registry.json')
    assert len(result['forecast'])==28 and np.isfinite(result['forecast']).all()
    assert result['metadata']['training_population_contains_series'] is False

def test_all_generalizable_cutoffs_historically_valid():
    registry=json.loads((ROOT/'models/generalizable_lightgbm/registry.json').read_text())
    for entry in registry['models'].values():
        assert set(entry['snapshots'])=={'1801','1829','1857','1885'}
        assert all(s['trained_through_day']<=int(origin) for origin,s in entry['snapshots'].items())
    metadata=json.loads((ROOT/'models/generalizable_lightgbm/origin_1885.metadata.json').read_text())
    assert metadata['random_seed']==42 and metadata['mode']=='pilot'
