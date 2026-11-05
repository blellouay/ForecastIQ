"""Leakage-safe *per-series* rolling validation for lightweight statistical routing.

No learned meta-router and no deep learning. A model is chosen using strictly
historical validation windows ending at or before the requested forecast origin.
This is an operational router, NOT proof a model is globally optimal.
"""
from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
from src.agent_tools.forecasting_tools import STATISTICAL, ToolError, _history, _horizon, forecast_statistical


def demand_profile(history: Sequence[float]) -> dict[str, Any]:
    """ADI and CV^2 using positive demand only; first observed sale is not a launch date."""
    a = _history(history)
    pos_indices = np.flatnonzero(a > 0)
    if len(pos_indices) < 2:
        return {'adi': None, 'cv2': None, 'demand_type': 'Insufficient history',
                'demand_type_note': 'Need >=2 positive sales; launch and availability unknown.'}
    # Active-history proxy: from first recorded sale (not verified launch) to cutoff.
    active = a[pos_indices[0]:]
    positives = active[active > 0]
    adi = float(len(active) / len(positives))
    cv2 = float(np.var(positives, ddof=0) / (np.mean(positives)**2))
    intermittent = adi >= 1.32
    variable = cv2 >= 0.49
    kind = ('Lumpy' if variable else 'Intermittent') if intermittent else ('Erratic' if variable else 'Smooth')
    return {'adi': adi, 'cv2': cv2, 'demand_type': kind,
            'demand_type_note': 'Classification uses observed sales from first positive sale, not verified launch.'}


def validation_cutoffs(n_days: int, horizon: int = 28, n_folds: int = 3,
                       min_train_days: int = 90) -> list[int]:
    """Return strictly historical fold cutoffs (0-based exclusive training ends).

    Outer target observations are never used in scoring: last validation target
    ends at n_days, and model is finally refit with all n_days of history.
    """
    horizon = _horizon(horizon)
    for label, val in [('n_folds', n_folds), ('min_train_days', min_train_days)]:
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ToolError(f'{label} must be a positive integer')
    if n_folds > 12:
        raise ToolError('n_folds must be <= 12 to bound computation')
    cutoffs = [n_days - horizon * k for k in range(n_folds, 0, -1)]
    if not cutoffs or cutoffs[0] < min_train_days:
        return []
    return cutoffs


def select_by_rolling_validation(series_id: str, sales_history: Sequence[float],
                                 horizon: int = 28, alpha: float = 0.2,
                                 n_folds: int = 3, min_train_days: int = 90,
                                 candidates: Sequence[str] = STATISTICAL) -> dict[str, Any]:
    """Select lowest mean MAE across *identical* historical folds.

    Candidate tuning is intentionally limited to predeclared alpha. Does not use
    older benchmark aggregate metrics or labels from future evaluation periods.
    """
    history = _history(sales_history)
    h = _horizon(horizon)
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not (0 < alpha <= 1):
        raise ToolError('alpha must be finite and in (0, 1]')
    if not candidates or len(set(candidates)) != len(candidates) or any(m not in STATISTICAL for m in candidates):
        raise ToolError('Candidates must be distinct supported statistical models')
    cutoffs = validation_cutoffs(len(history), h, n_folds, min_train_days)
    if not cutoffs:
        return {'model': 'TSB', 'method': 'fallback', 'fold_cutoffs': [],
                'reason': f'TSB conservative fallback: insufficient history for {n_folds} historical {h}-day validation folds; no comparative claim.',
                'validation_mae': {}, 'alpha': float(alpha)}
    scores: dict[str, list[float]] = {}
    failures: dict[str, str] = {}
    for model in candidates:
        errors = []
        try:
            for cutoff in cutoffs:
                pred = forecast_statistical(series_id, history[:cutoff], model=model, horizon=h, alpha=alpha)['forecast']
                actual = history[cutoff:cutoff+h]
                if len(actual) != h:
                    raise ToolError('Validation target window length mismatch')
                errors.append(float(np.abs(np.asarray(pred) - actual).mean()))
            scores[model] = errors
        except (ToolError, ValueError, TypeError) as exc:
            failures[model] = str(exc)
    if not scores:
        raise ToolError(f'All candidate models failed historical validation: {failures}')
    # Fixed, documented tie order; no tie-breaking on evaluation outcomes.
    preference = ('TSB','SES','SBA','Croston','Average','SeasonalNaive','Naive')
    ranked = sorted(scores, key=lambda m: (np.mean(scores[m]), preference.index(m)))
    selected = ranked[0]
    means = {k: float(np.mean(v)) for k,v in scores.items()}
    return {'model': selected, 'method': 'historical_rolling_mae',
            'fold_cutoffs': cutoffs, 'fold_mae': scores,
            'validation_mae': means, 'failures': failures,
            'alpha': float(alpha), 'n_folds': len(cutoffs),
            'reason': (f'{selected} selected by lowest mean historical MAE across {len(cutoffs)} '
                       f'non-overlapping {h}-day validation folds ({means[selected]:.4f}). '
                       'Selection is local to this series; no global superiority claim.')}
