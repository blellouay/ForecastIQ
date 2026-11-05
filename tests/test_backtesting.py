import numpy as np
from src.evaluation.backtesting import expanding_origins,backtest_series
def test_nonoverlapping_origins_and_no_future_access():
    origins=expanding_origins(300,28,6); assert np.all(np.diff(origins)==28)
    calls=[]
    def predictor(history,h): calls.append(len(history)); return np.zeros(h)
    rows=backtest_series(np.ones(300),predictor,origins)
    assert calls==origins and len(rows)==168
