import math
import pandas as pd
import pytest
from dashboard.logic import (parse_m5_id, make_request, forecast_table, selection_frame,
                             extract_ai_part, filter_catalog, model_availability_frame)

@pytest.mark.parametrize('identifier,expected', [
    ('FOODS_1_001_CA_1_validation', ('FOODS_1_001','CA_1')),
    ('HOBBIES_2_015_WI_3_evaluation', ('HOBBIES_2_015','WI_3')),
    ('HOUSEHOLD_1_123_TX_2_validation', ('HOUSEHOLD_1_123','TX_2')),
])
def test_id(identifier, expected):
    assert parse_m5_id(identifier) == expected

def test_reject_unknown_id():
    with pytest.raises(ValueError): parse_m5_id('FOODS_1_001__CA_1')

def test_request_model_explicit():
    assert make_request('FOODS_1_001','CA_1','NHITS_small').endswith('using NHITS_small')

def test_request_bad_horizon():
    with pytest.raises(ValueError): make_request('FOODS_1_001','CA_1','auto',7)

def test_forecast_table():
    data = forecast_table({'status':'completed','forecast_result':{'forecast':[.5]*28}},1857)
    assert len(data) == 28 and data.iloc[0]['day'] == 1858
    assert math.isclose(data['forecast'].sum(),14.0)

def test_forecast_invalid():
    with pytest.raises(ValueError): forecast_table({'status':'completed','forecast_result':{'forecast':[float('nan')]*28}},1857)

def test_sorted_validation():
    data = selection_frame({'selection_evidence':{'validation_mae':{'TSB':0.7,'SBA':0.6}}})
    assert data.iloc[0]['Model'] == 'SBA'

def test_comment():
    text, status = extract_ai_part('Forecast.\nAI interpretation (non-numeric): Explanation', {})
    assert text == 'Explanation' and status == 'success'

def test_model_availability_table():
    frame = model_availability_frame({'model_availability':[
        {'model':'NHITS_small','family':'deep_learning','inference_eligible':True,
         'selection_eligible':False,'reason':'Inference only'}]})
    assert frame.iloc[0].to_dict() == {
        'Model':'NHITS_small','Family':'deep_learning','Inference':'Yes',
        'Auto selection':'No','Reason':'Inference only'}

def test_lightgbm_catalog_filter(tmp_path):
    mapping = tmp_path / 'mapping.json'
    mapping.write_text('{"series_id_to_code":{"FOODS_1_001__CA_1":7}}', encoding='utf-8')
    catalog = {'FOODS_1_001':['CA_1','TX_1'], 'FOODS_1_002':['CA_1']}
    assert filter_catalog(catalog, 'LightGBM-compatible products', str(mapping)) == {
        'FOODS_1_001':['CA_1']}
