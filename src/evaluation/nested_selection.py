"""Leakage-free configuration selection for nested rolling-origin studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def eligible_inner_origins(outer_origin: int, *, horizon: int = 28,
                           n_folds: int = 3, min_history: int = 112,
                           step: int | None = None) -> tuple[int, ...]:
    """Return the latest non-overlapping folds whose targets end by outer origin."""
    if horizon < 1 or n_folds < 1:
        raise ValueError("horizon and n_folds must be positive")
    step = horizon if step is None else int(step)
    candidates = [outer_origin - horizon - i * step for i in range(n_folds)]
    origins = tuple(sorted(origin for origin in candidates if origin >= min_history))
    if not origins:
        raise ValueError("outer origin has no eligible inner validation folds")
    if any(origin + horizon > outer_origin for origin in origins):
        raise AssertionError("inner validation target extends beyond outer origin")
    return origins


def stable_config_id(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def aggregate_validation(validation: pd.DataFrame, *, expected_origins: Iterable[int],
                         metric: str = "MAE") -> pd.DataFrame:
    """Aggregate only configurations successful on every expected fold."""
    required = {"family", "config_id", "inner_origin", metric, "status"}
    missing = required - set(validation)
    if missing:
        raise ValueError(f"validation table missing columns: {sorted(missing)}")
    expected = tuple(sorted(map(int, expected_origins)))
    valid = validation[(validation.status == "completed") & np.isfinite(validation[metric])].copy()
    rows = []
    for (family, config_id), group in valid.groupby(["family", "config_id"], sort=True):
        observed = tuple(sorted(group.inner_origin.astype(int).unique()))
        complete = observed == expected and not group.inner_origin.duplicated().any()
        rows.append({"family": family, "config_id": config_id,
                     "mean_MAE": float(group[metric].mean()) if complete else np.nan,
                     "std_MAE": float(group[metric].std(ddof=1)) if complete and len(group) > 1 else 0.0,
                     "n_folds": int(group.inner_origin.nunique()),
                     "expected_folds": len(expected), "complete": bool(complete),
                     "inner_origins": list(observed)})
    return pd.DataFrame(rows)


def select_configurations(validation: pd.DataFrame, *, expected_origins: Iterable[int],
                          metric: str = "MAE") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select lowest mean loss per family with deterministic config-id ties."""
    leaderboard = aggregate_validation(validation, expected_origins=expected_origins, metric=metric)
    complete = leaderboard[leaderboard.complete & leaderboard.mean_MAE.notna()].copy()
    if complete.empty:
        raise ValueError("no configuration completed all expected validation folds")
    selected = (complete.sort_values(["family", "mean_MAE", "config_id"], kind="stable")
                .groupby("family", as_index=False, sort=True).first())
    selected["selection_criterion"] = f"minimum mean {metric} across all eligible inner folds"
    return selected, leaderboard.sort_values(["family", "complete", "mean_MAE", "config_id"],
                                              ascending=[True, False, True, True], na_position="last")


def assert_nested_history(selection_history: pd.DataFrame, *, horizon: int = 28) -> None:
    required = {"outer_origin", "inner_origin"}
    if not required.issubset(selection_history):
        raise ValueError(f"selection history missing {sorted(required-set(selection_history))}")
    if ((selection_history.inner_origin.astype(int) + horizon) >
            selection_history.outer_origin.astype(int)).any():
        raise ValueError("look-ahead bias: inner validation ends after outer origin")


class ValidationCache:
    """Small resumable JSON cache keyed by data/configuration/protocol hashes."""
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def load(self, key: dict[str, Any]) -> dict[str, Any] | None:
        path = self.path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self, key: dict[str, Any], result: dict[str, Any]) -> Path:
        path = self.path(key)
        path.write_text(json.dumps({"key": key, "result": result}, indent=2, default=str), encoding="utf-8")
        return path
