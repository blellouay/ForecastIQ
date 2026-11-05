"""Origin-safe Syntetos-Boylan-Croston demand classification."""
from __future__ import annotations
import numpy as np
from .preparation import apply_leading_zero_policy

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def demand_statistics(values, *, leading_zero_policy="active", minimum_positive=2):
    y = apply_leading_zero_policy(values, leading_zero_policy)
    positive = y[y > 0]
    if positive.size == 0:
        return {"ADI": np.nan, "CV2": np.nan, "positive_days": 0, "active_days": 0}
    adi = y.size / positive.size
    cv2 = np.var(positive, ddof=0) / positive.mean() ** 2
    return {"ADI": float(adi), "CV2": float(cv2), "positive_days": int(positive.size),
            "active_days": int(y.size), "insufficient": positive.size < minimum_positive}


def classify_demand(values, *, adi_threshold=ADI_THRESHOLD, cv2_threshold=CV2_THRESHOLD,
                    leading_zero_policy="active", minimum_positive=2):
    stats = demand_statistics(values, leading_zero_policy=leading_zero_policy,
                              minimum_positive=minimum_positive)
    if stats["positive_days"] == 0:
        return "No sales"
    if stats["insufficient"]:
        return "Insufficient history"
    low_adi, low_cv2 = stats["ADI"] < adi_threshold, stats["CV2"] < cv2_threshold
    return ("Smooth" if low_cv2 else "Erratic") if low_adi else ("Intermittent" if low_cv2 else "Lumpy")
