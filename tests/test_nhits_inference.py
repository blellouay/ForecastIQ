from datetime import date,timedelta
import pandas as pd
import pytest
from unittest.mock import patch
from src.agents.nhits_inference import (predict_nhits, predict_nhits_scalable,
    scalable_checkpoint_path, NHITSInferenceError, _canonical)
from src.agents.workflow import select_model, generate_forecast


def fake_predictions():
    start=date(2016,2,29)
    return pd.DataFrame([{'unique_id':'FOODS_1_008__TX_1','ds':start+timedelta(days=i),'NHITS_small':float(i)/100}
                         for i in range(28)])


def test_id_normalization():
    assert _canonical('FOODS_1_008_TX_1_validation') == 'FOODS_1_008__TX_1'
    assert _canonical('FOODS_1_008__TX_1') == 'FOODS_1_008__TX_1'


def test_checkpoint_predict_mock():
    with patch('src.agents.nhits_inference._predict_all', return_value=fake_predictions()):
        result=predict_nhits('FOODS_1_008_TX_1_validation',1857)
        assert len(result['forecast']) == 28
        assert result['forecast'][1] == .01
        assert result['metadata']['forecast_start_date']=='2016-02-29'


def test_reject_wrong_origin():
    with patch('src.agents.nhits_inference._predict_all', return_value=fake_predictions()):
        with pytest.raises(NHITSInferenceError,match='not the requested'):
            predict_nhits('FOODS_1_008__TX_1',1885)


def test_reject_unknown_series():
    with patch('src.agents.nhits_inference._predict_all', return_value=fake_predictions()):
        with pytest.raises(NHITSInferenceError,match='not supported'):
            predict_nhits('FOODS_1_001__CA_1',1857)


def test_reject_horizon():
    with pytest.raises(NHITSInferenceError,match='28-day'):
        predict_nhits('FOODS_1_008__TX_1',1857,7)


def test_explicit_selection():
    state={'model':'NHITS_small','forecast_origin_day':1857,'series_id':'FOODS_1_008_TX_1_validation'}
    selected=select_model(state)
    assert selected['selected_model']=='NHITS_small'


def test_auto_not_add_nhits():
    # NHITS must NOT be included as a historical candidate by this patch.
    from src.agents.ml_routing import ML_NAMES
    assert 'NHITS_small' not in ML_NAMES


def test_supplied_history_unseen_series_and_date_alignment():
    class FakeNF:
        uids = pd.Index(['FOODS_1_008__TX_1'])
        def predict(self, df):
            start = pd.to_datetime(df.ds.max()) + pd.Timedelta(days=1)
            return pd.DataFrame({
                'unique_id':[df.unique_id.iloc[0]] * 28,
                'ds':pd.date_range(start, periods=28, freq='D'),
                'NHITS_small':[0.25] * 28,
            })
    with patch('src.agents.nhits_inference._load', return_value=FakeNF()):
        result = predict_nhits('FOODS_1_001_CA_1_validation', 1885,
                               historical_sales=[0.0] * 1885)
    metadata = result['metadata']
    assert metadata['forecast_start_date'] == '2016-03-28'
    assert metadata['forecast_end_date'] == '2016-04-24'
    assert metadata['historical_observations'] == 1885
    assert metadata['history_source'] == 'caller-supplied origin-truncated M5 history'
    assert metadata['inference_device'] == 'cpu'
    assert metadata['series_in_training'] is False
    assert result['forecast'] == [0.25] * 28


def test_supplied_history_rejects_origin_before_training_cutoff():
    with pytest.raises(NHITSInferenceError, match='trained through d_1857'):
        predict_nhits('FOODS_1_001_CA_1_validation', 1829,
                      historical_sales=[0.0] * 1829)


@pytest.mark.integration
def test_scalable_checkpoint_provenance_and_origin():
    path = scalable_checkpoint_path('models/nhits_full_m5/full_all', 1801)
    assert path.name == 'origin_1801'
    with pytest.raises(NHITSInferenceError, match='No NHITS_scalable snapshot'):
        scalable_checkpoint_path('models/nhits_full_m5/full_all', 1810)


@pytest.mark.integration
def test_scalable_supplied_history_dates_and_nonnegative_policy():
    class FakeNF:
        def predict(self, df):
            return pd.DataFrame({
                'unique_id': [df.unique_id.iloc[0]] * 28,
                'ds': pd.date_range(pd.to_datetime(df.ds.max()) + pd.Timedelta(days=1), periods=28),
                'NHITS_scalable': [-1.0] + [0.5] * 27,
            })
    with patch('src.agents.nhits_inference._load_scalable', return_value=FakeNF()):
        result = predict_nhits_scalable('FOODS_1_001_CA_1_validation', 1801,
                                        [0.0] * 1801)
    assert result['forecast'][0] == 0.0
    assert result['metadata']['negative_predictions_clipped'] == 1
    assert result['metadata']['forecast_start_date'] == '2016-01-04'
    assert result['metadata']['training_cutoff_day'] == 1801
    assert result['metadata']['unseen_generalization_claim'] is False


@pytest.mark.integration
def test_scalable_rejects_history_not_through_origin():
    with pytest.raises(NHITSInferenceError, match='exactly d_1'):
        predict_nhits_scalable('FOODS_1_001_CA_1_validation', 1801, [0.0] * 1800)
