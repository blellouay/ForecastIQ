from src.forecasting.neural_models import FAST_MODE, RESEARCH_MODE, neural_candidates, training_mode


def test_research_mode_is_not_smoke_configuration():
    assert FAST_MODE.max_steps == 20
    assert 500 <= RESEARCH_MODE.max_steps <= 1000
    assert len(RESEARCH_MODE.validation_origins) >= 3
    assert RESEARCH_MODE.batch_size <= FAST_MODE.batch_size


def test_candidates_include_both_deepar_distributions_and_small_nhits():
    candidates = neural_candidates("research")
    assert {c.get("distribution") for c in candidates if c["family"] == "DeepAR"} == {
        "StudentT", "NegativeBinomial"
    }
    assert any(c["name"] == "NHITS_small" and c["mlp_units"] for c in candidates)
    assert training_mode("FAST") == FAST_MODE
