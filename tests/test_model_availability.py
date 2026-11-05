import pytest
from src.agents.model_availability import assess_model_availability


@pytest.mark.integration
def test_inference_and_selection_are_distinct():
    rows = assess_model_availability(
        'FOODS_1_001_CA_1_validation', [0.0] * 1885, 1885,
        nhits_checkpoint_path='models/neuralforecast/nhits_small')
    by_name = {row['model']: row for row in rows}
    assert by_name['SBA']['inference_eligible'] is True
    assert by_name['SBA']['selection_eligible'] is True
    assert by_name['NHITS_small']['inference_eligible'] is True
    assert by_name['NHITS_small']['selection_eligible'] is False
    assert by_name['NHITS_scalable']['inference_eligible'] is True
    assert by_name['NHITS_scalable']['selection_eligible'] is True
    assert by_name['DeepAR']['inference_eligible'] is False
    assert by_name['DeepAR']['selection_eligible'] is False


def test_deepar_explicit_selection_fails_truthfully():
    from src.agents.workflow import select_model
    result = select_model({'model':'DeepAR'})
    assert result['status'] == 'error'
    assert 'no verified deployable inference bundle' in result['error']
