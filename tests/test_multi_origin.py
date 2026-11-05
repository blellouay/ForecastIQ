import pandas as pd
import pytest

from src.evaluation.multi_origin import (assert_complete_matched_backtest, coverage_table,
                                         rolling_training_origins)


def _records(models=("a", "b"), origins=(10, 12)):
    rows = []
    for model in models:
        for origin in origins:
            for series in ("s1", "s2"):
                for horizon in (1, 2):
                    rows.append({"model": model, "forecast_origin": origin,
                                 "series_id": series, "horizon": horizon,
                                 "target_date": origin + horizon,
                                 "actual": 1.0, "prediction": 1.0})
    return pd.DataFrame(rows)


def test_complete_matched_backtest_passes():
    records = _records()
    assert_complete_matched_backtest(records, ("a", "b"), (10, 12),
                                     series_count=2, horizon=2)
    assert set(coverage_table(records, ("a", "b"), (10, 12),
                              series_count=2, horizon=2).status) == {"complete"}


def test_partial_model_cannot_enter_complete_benchmark():
    records = _records().iloc[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_matched_backtest(records, ("a", "b"), (10, 12),
                                         series_count=2, horizon=2)


def test_rolling_training_origins_are_chronological_and_non_overlapping():
    origins = rolling_training_origins(200, horizon=10, folds=4)
    assert origins == (160, 170, 180, 190)
    assert all(origin + 10 <= 200 for origin in origins)
