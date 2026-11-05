"""Reference statistical forecasts used by every ForecastIQ experiment.

Conventions
-----------
Inputs are non-negative, finite daily observations. ``leading_zero_policy`` is
``"preserve"`` by default; ``"active"`` trims observations before the first
positive sale (an observed-sales proxy, not a known launch date). Croston and
SBA initialize demand size at the first event and the interval at one, then
smooth completed inter-demand intervals. TSB initializes probability at one
on the first event and updates probability on every subsequent day. SES starts
at the first observation. Every method returns a finite non-negative vector.
"""
from __future__ import annotations

import numpy as np


def _history(values, leading_zero_policy="preserve"):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("history must be a non-empty one-dimensional array")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("history must contain finite non-negative values")
    if leading_zero_policy not in {"preserve", "active"}:
        raise ValueError("leading_zero_policy must be 'preserve' or 'active'")
    if leading_zero_policy == "active":
        hits = np.flatnonzero(y > 0)
        if hits.size:
            y = y[hits[0] :]
    return y


def _horizon(horizon):
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


def _alpha(value, name="alpha"):
    value = float(value)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def _constant(rate, horizon):
    rate = float(np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0))
    return np.full(_horizon(horizon), max(rate, 0.0), dtype=float)


def forecast_average(history, horizon, *, leading_zero_policy="preserve"):
    y = _history(history, leading_zero_policy)
    return _constant(y.mean(), horizon)


def forecast_naive(history, horizon, *, leading_zero_policy="preserve"):
    y = _history(history, leading_zero_policy)
    return _constant(y[-1], horizon)


def forecast_seasonal_naive(history, horizon, *, season_length=7, leading_zero_policy="preserve"):
    y = _history(history, leading_zero_policy)
    season_length = int(season_length)
    if season_length < 1:
        raise ValueError("season_length must be positive")
    pattern = y[-min(season_length, y.size) :]
    return np.resize(pattern, _horizon(horizon)).astype(float)


def forecast_ses(history, horizon, *, alpha=0.2, leading_zero_policy="preserve"):
    y = _history(history, leading_zero_policy)
    alpha = _alpha(alpha)
    level = float(y[0])
    for value in y[1:]:
        level += alpha * (float(value) - level)
    return _constant(level, horizon)


def _croston_rate(history, alpha, correction, leading_zero_policy):
    y = _history(history, leading_zero_policy)
    alpha = _alpha(alpha)
    hits = np.flatnonzero(y > 0)
    if not hits.size:
        return 0.0
    size = float(y[hits[0]])
    interval = 1.0
    last = int(hits[0])
    for index in hits[1:]:
        index = int(index)
        size += alpha * (float(y[index]) - size)
        interval += alpha * ((index - last) - interval)
        last = index
    return correction(alpha) * size / interval


def forecast_croston(history, horizon, *, alpha=0.2, leading_zero_policy="preserve"):
    return _constant(_croston_rate(history, alpha, lambda _: 1.0, leading_zero_policy), horizon)


def forecast_sba(history, horizon, *, alpha=0.2, leading_zero_policy="preserve"):
    return _constant(_croston_rate(history, alpha, lambda a: 1.0 - a / 2.0, leading_zero_policy), horizon)


def forecast_tsb(history, horizon, *, alpha=0.2, beta=None, leading_zero_policy="preserve"):
    y = _history(history, leading_zero_policy)
    alpha = _alpha(alpha)
    beta = _alpha(alpha if beta is None else beta, "beta")
    hits = np.flatnonzero(y > 0)
    if not hits.size:
        return _constant(0.0, horizon)
    first = int(hits[0])
    size, probability = float(y[first]), 1.0
    for value in y[first + 1 :]:
        event = float(value > 0)
        probability += beta * (event - probability)
        if event:
            size += alpha * (float(value) - size)
    return _constant(probability * size, horizon)


MODELS = {
    "Average": forecast_average, "Naive": forecast_naive,
    "SeasonalNaive": forecast_seasonal_naive, "SES": forecast_ses,
    "Croston": forecast_croston, "SBA": forecast_sba, "TSB": forecast_tsb,
}


def forecast(model, history, horizon, **kwargs):
    try:
        function = MODELS[model]
    except KeyError as exc:
        raise ValueError(f"unknown model: {model}") from exc
    return function(history, horizon, **kwargs)


def forecast_matrix(model, histories, horizon, **kwargs):
    """Vectorized panel API whose values match the scalar reference exactly."""
    matrix = np.asarray(histories, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("histories must be a two-dimensional array")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or matrix.shape[1] == 0:
        raise ValueError("histories must be non-empty, finite and non-negative")
    if kwargs.get("leading_zero_policy", "preserve") != "preserve":
        return np.vstack([forecast(model, row, horizon, **kwargs) for row in matrix])
    n, length = matrix.shape; horizon = _horizon(horizon)
    if model == "Average": rates = matrix.mean(1)
    elif model == "Naive": rates = matrix[:, -1]
    elif model == "SeasonalNaive":
        season = int(kwargs.get("season_length", 7))
        if season < 1: raise ValueError("season_length must be positive")
        pattern = matrix[:, -min(season, length):]
        return pattern[:, np.arange(horizon) % pattern.shape[1]].copy()
    elif model == "SES":
        alpha=_alpha(kwargs.get("alpha",.2)); rates=matrix[:,0].copy()
        for day in range(1,length): rates += alpha*(matrix[:,day]-rates)
    elif model in {"Croston","SBA","TSB"}:
        alpha=_alpha(kwargs.get("alpha",.2)); positive=matrix>0; any_positive=positive.any(1)
        first=np.argmax(positive,axis=1); size=np.where(any_positive,matrix[np.arange(n),first],0.).astype(float)
        if model=="TSB":
            beta=_alpha(kwargs.get("beta",alpha),"beta"); probability=np.where(any_positive,1.,0.)
            for day in range(length):
                valid=any_positive & (day>first); event=positive[:,day]
                probability[valid] += beta*(event[valid].astype(float)-probability[valid])
                update=valid & event; size[update] += alpha*(matrix[update,day]-size[update])
            rates=probability*size
        else:
            interval=np.ones(n); last=first.copy()
            for day in range(length):
                event=any_positive & positive[:,day] & (day>first)
                size[event] += alpha*(matrix[event,day]-size[event])
                interval[event] += alpha*((day-last[event])-interval[event]); last[event]=day
            rates=size/interval
            if model=="SBA": rates*=1-alpha/2
    else: raise ValueError(f"unknown model: {model}")
    rates=np.maximum(np.nan_to_num(rates,nan=0.,posinf=0.,neginf=0.),0.)
    return np.broadcast_to(rates[:,None],(n,horizon)).copy()
