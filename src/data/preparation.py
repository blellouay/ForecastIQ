"""Consistent daily-sales validation and active-history helpers."""
from __future__ import annotations
import numpy as np


def validate_daily_sales(values):
    y = np.asarray(values, dtype=float)
    if y.ndim not in (1, 2) or y.size == 0:
        raise ValueError("sales must be a non-empty 1D or 2D array")
    if not np.isfinite(y).all():
        raise ValueError("sales contain missing or non-finite values")
    if (y < 0).any():
        raise ValueError("sales contain negative values")
    return y


def first_positive_index(values):
    y = validate_daily_sales(values)
    if y.ndim != 1:
        raise ValueError("expected one series")
    hits = np.flatnonzero(y > 0)
    return int(hits[0]) if hits.size else None


def apply_leading_zero_policy(values, policy="preserve"):
    y = validate_daily_sales(values).copy()
    if y.ndim != 1:
        raise ValueError("expected one series")
    if policy == "preserve":
        return y
    if policy != "active":
        raise ValueError("policy must be 'preserve' or 'active'")
    first = first_positive_index(y)
    return y[first:] if first is not None else y


def active_history_length(values):
    y = validate_daily_sales(values)
    first = first_positive_index(y)
    return 0 if first is None else int(y.size - first)


def zero_sales_ratio(values, policy="preserve"):
    y = apply_leading_zero_policy(values, policy)
    return float(np.mean(y == 0))


def insufficient_history(values, minimum=112, policy="preserve"):
    return apply_leading_zero_policy(values, policy).size < int(minimum)
