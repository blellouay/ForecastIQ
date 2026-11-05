"""Lightweight fake artifacts for tests only; no retraining or real LightGBM required."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from src.agent_tools import ToolError
from src.agents.ml_routing import predict_snapshot, route_with_saved_ml
from src.agents.workflow import select_model, generate_forecast


def provider(series_id, history, origin, horizon, names):
    assert len(history) == origin  # synthetic origin day equals history length
    return pd.DataFrame({n: np.full(horizon, 0.0) for n in names})


class ConstantModel:
    def __init__(self, value): self.value = value
    def predict(self, x): return np.full(len(x), self.value)


class OccurrenceModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, x): return np.tile([0.25, 0.75], (len(x), 1))


def registry(tmp_path, total=140, invalid_cutoff=None):
    import joblib
    joblib.dump(ConstantModel(0.0), tmp_path/'direct.joblib')
    joblib.dump(OccurrenceModel(), tmp_path/'classifier.joblib')
    joblib.dump(ConstantModel(2.0), tmp_path/'quantity.joblib')
    snapshots = {}
    for origin in [140, 168, 196, 224]:
        snapshots[str(origin)] = {'forecast_origin_day': origin,
                                  'trained_through_day': invalid_cutoff if invalid_cutoff and origin == 140 else origin,
                                  'feature_pipeline_id':'pipeline_v1',
                                  'model_path':'direct.joblib',
                                  'classifier_path':'classifier.joblib',
                                  'regressor_path':'quantity.joblib'}
    doc = {'schema_version':1,'models': {
        'DirectLightGBM':{'kind':'direct','feature_names':['lag_7'],'feature_pipeline_id':'pipeline_v1','snapshots':snapshots},
        'HurdleLightGBM':{'kind':'hurdle','feature_names':['lag_7'],'feature_pipeline_id':'pipeline_v1','snapshots':snapshots}}}
    path = tmp_path/'registry.json'; path.write_text(json.dumps(doc)); return path


def test_direct_and_hurdle_infer(tmp_path):
    path = registry(tmp_path)
    a = predict_snapshot('A', [0.0]*224,224,28,'DirectLightGBM',path,provider)
    b = predict_snapshot('A', [0.0]*224,224,28,'HurdleLightGBM',path,provider)
    assert a['total_expected_demand'] == 0
    assert b['total_expected_demand'] == pytest.approx(42)


def test_future_cutoff_rejected(tmp_path):
    path = registry(tmp_path, invalid_cutoff=225)
    with pytest.raises(ToolError, match='training cutoff'):
        predict_snapshot('A',[0]*140,140,28,'DirectLightGBM',path,provider)


def test_unregistered_or_stale_model_fails_closed(tmp_path):
    path = registry(tmp_path)
    with pytest.raises(ToolError, match='No as-of snapshot'):
        predict_snapshot('A',[0]*223,223,28,'DirectLightGBM',path,provider)
    with pytest.raises(ToolError, match='not registered'):
        predict_snapshot('A',[0]*224,224,28,'OtherModel',path,provider)


def test_incomplete_ml_folds_excluded_but_stats_work(tmp_path):
    path = registry(tmp_path)
    doc=json.loads(path.read_text())
    del doc['models']['DirectLightGBM']['snapshots']['168']
    path.write_text(json.dumps(doc))
    with patch('src.agents.ml_routing.select_by_rolling_validation') as stat:
        stat.return_value={'model':'TSB','reason':'statistically routed','validation_mae':{'TSB':2.},
                           'fold_mae':{'TSB':[2.,2.,2.]}, 'alpha':0.2}
        result=route_with_saved_ml('A',[0]*224,224,28,path,provider)
    assert 'DirectLightGBM' in result['ml_excluded']
    assert result['model']=='HurdleLightGBM'


def test_no_registry_keeps_statistical_selection(tmp_path):
    with patch('src.agents.ml_routing.select_by_rolling_validation') as stat:
        stat.return_value={'model':'SBA','reason':'selected by stat CV','validation_mae':{'SBA':0.6},'fold_mae':{'SBA':[0.6]*3}}
        result=route_with_saved_ml('A',[0]*224,224,28,None,None)
    assert result['model']=='SBA' and len(result['ml_excluded'])==2


def test_no_provider_manual_ml_error():
    x=select_model({'model':'DirectLightGBM','ml_registry_path':'models.json'})
    assert x['status']=='error'


def test_future_sales_cannot_influence_fold_features(tmp_path):
    path=registry(tmp_path)
    observed=[]
    def track(series, history, origin, horizon, names):
        observed.append((origin, tuple(history)))
        return provider(series, history, origin, horizon, names)
    with patch('src.agents.ml_routing.select_by_rolling_validation') as stat:
        stat.return_value={'model':'TSB','reason':'test','validation_mae':{'TSB':2.},'fold_mae':{'TSB':[2.]*3}}
        route_with_saved_ml('A',[0.]*224,224,28,path,track)
    assert set(origin for origin,_ in observed)=={140,168,196}
    assert all(len(hist)==origin for origin,hist in observed)
