import numpy as np
import pandas as pd
import pytest

from src.evaluation.nested_selection import (
    ValidationCache, aggregate_validation, assert_nested_history,
    eligible_inner_origins, select_configurations, stable_config_id,
)
from src.forecasting.statistical_models import forecast


def _validation(rows):
    return pd.DataFrame(rows, columns=["family", "config_id", "inner_origin", "MAE", "status"])


def test_inner_targets_end_no_later_than_outer_origin():
    inner = eligible_inner_origins(1717, horizon=28, n_folds=3)
    assert inner == (1633, 1661, 1689)
    assert all(origin + 28 <= 1717 for origin in inner)
    history = pd.DataFrame({"outer_origin": [1717] * 3, "inner_origin": inner})
    assert_nested_history(history, horizon=28)
    history.loc[0, "inner_origin"] = 1700
    with pytest.raises(ValueError, match="look-ahead"):
        assert_nested_history(history, horizon=28)


def test_selection_uses_fold_mean_not_best_isolated_fold():
    rows = []
    for origin, loss in zip((10, 20, 30), (0.0, 10.0, 10.0)):
        rows.append(("DL", "isolated", origin, loss, "completed"))
    for origin, loss in zip((10, 20, 30), (4.0, 4.0, 4.0)):
        rows.append(("DL", "consistent", origin, loss, "completed"))
    selected, _ = select_configurations(_validation(rows), expected_origins=(10, 20, 30))
    assert selected.iloc[0].config_id == "consistent"


def test_configuration_missing_a_fold_is_ineligible():
    rows = [("DL", "partial", 10, 0.1, "completed"),
            ("DL", "partial", 20, 0.1, "completed")]
    rows += [("DL", "complete", o, 1.0, "completed") for o in (10, 20, 30)]
    selected, leaderboard = select_configurations(_validation(rows), expected_origins=(10, 20, 30))
    assert selected.iloc[0].config_id == "complete"
    assert not leaderboard.loc[leaderboard.config_id == "partial", "complete"].item()


def _select_ses(values, outer):
    rows = []
    inner = eligible_inner_origins(outer, horizon=7, n_folds=3, min_history=30)
    for alpha in (0.1, 0.5, 0.9):
        config = stable_config_id({"model": "SES", "alpha": alpha})
        for origin in inner:
            pred = forecast("SES", values[:origin], 7, alpha=alpha)
            rows.append(("SES", config, origin,
                         np.abs(pred-values[origin:origin+7]).mean(), "completed"))
    return select_configurations(_validation(rows), expected_origins=inner)[0].config_id.item()


def test_changing_demand_after_outer_origin_cannot_change_selection():
    rng = np.random.default_rng(42)
    values = rng.poisson(2.0, 150).astype(float)
    outer = 100
    changed = values.copy(); changed[outer:] = 10000
    assert _select_ses(values, outer) == _select_ses(changed, outer)


def test_cache_key_is_deterministic_and_resumable(tmp_path):
    cache = ValidationCache(tmp_path)
    key = {"outer": 100, "seed": 42, "config": {"alpha": 0.2}}
    assert cache.load(key) is None
    first = cache.save(key, {"MAE": 1.5})
    second = cache.path({"config": {"alpha": 0.2}, "seed": 42, "outer": 100})
    assert first == second
    assert cache.load(key)["result"]["MAE"] == 1.5


def test_duplicate_fold_is_not_complete():
    validation = _validation([("x", "a", 10, 1.0, "completed"),
                              ("x", "a", 10, 1.0, "completed"),
                              ("x", "a", 20, 1.0, "completed")])
    board = aggregate_validation(validation, expected_origins=(10, 20))
    assert not board.complete.item()
