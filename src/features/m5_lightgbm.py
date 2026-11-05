"""Trusted M5 feature provider for the 400-series LightGBM snapshots.

The schema intentionally mirrors notebooks 03 and 03B.  ``series_code`` is
loaded from the exported training mapping; it is never inferred from input
order.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_tools.forecasting_tools import ToolError
from .feature_engineering import historical_snapshot

FEATURE_NAMES = [
    "series_code", "horizon", "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",
    "mean_7", "std_7", "zero_ratio_7", "mean_28", "std_28", "zero_ratio_28",
    "mean_56", "std_56", "zero_ratio_56", "last_28_vs_prev_28",
    "day_of_week", "month",
]
PIPELINE_ID = "m5-notebook03-historical-snapshot-v1"
CANONICAL_ID_RE = re.compile(
    r"^(?P<item>(?:FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+)__(?P<store>(?:CA|TX|WI)_\d+)$"
)
FULL_M5_ID_RE = re.compile(
    r"^(?P<item>(?:FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+)_(?P<store>(?:CA|TX|WI)_\d+)_(?P<split>validation|evaluation)$"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_series_id(series_id: str) -> str:
    """Return the notebook's canonical ``ITEM_ID__STORE_ID`` identity.

    Only the two known M5 forms are accepted. This never assigns a code; the
    normalized identity must still be present in the persisted training map.
    """
    if not isinstance(series_id, str) or not series_id:
        raise ToolError("M5 series_id must be a nonempty string")
    canonical = CANONICAL_ID_RE.fullmatch(series_id)
    if canonical:
        return series_id
    full = FULL_M5_ID_RE.fullmatch(series_id)
    if full:
        return f"{full.group('item')}__{full.group('store')}"
    raise ToolError(f"Unsupported M5 series_id format: {series_id!r}")


def _load_mapping(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ToolError(f"LightGBM series mapping missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ToolError(f"LightGBM series mapping is invalid: {path}") from exc
    if document.get("feature_pipeline_id") != PIPELINE_ID:
        raise ToolError("LightGBM series mapping has an incompatible feature pipeline")
    ids = document.get("training_series_ids")
    mapping = document.get("series_id_to_code")
    if (not isinstance(ids, list) or not isinstance(mapping, dict) or
            mapping != {series_id: code for code, series_id in enumerate(ids)}):
        raise ToolError("LightGBM series mapping/order is inconsistent")
    canonical_to_code = {}
    for saved_id, code in mapping.items():
        canonical = normalize_series_id(saved_id)
        previous = canonical_to_code.setdefault(canonical, code)
        if previous != code:
            raise ToolError(
                f"Conflicting LightGBM mappings for canonical identity {canonical!r}"
            )
    document["canonical_series_id_to_code"] = canonical_to_code
    return document


def _calendar_dates(path: Path, origin: int, horizon: int) -> pd.DatetimeIndex:
    calendar = pd.read_csv(path, usecols=["d", "date"]).drop_duplicates("d").set_index("d")
    labels = [f"d_{day}" for day in range(origin + 1, origin + horizon + 1)]
    dates = pd.to_datetime(calendar.reindex(labels)["date"])
    if dates.isna().any():
        raise ToolError("M5 calendar lacks dates required by the forecast horizon")
    return pd.DatetimeIndex(dates.to_numpy())


def make_feature_provider(
    mapping_path: str | Path | None = None,
    calendar_path: str | Path | None = None,
):
    """Build the Step 4 provider with explicit, testable artifact locations."""
    root = _project_root()
    mapping_file = Path(mapping_path or root / "models/lightgbm/series_mapping.json").resolve()
    calendar_file = Path(calendar_path or root / "data/raw/calendar.csv").resolve()

    def provider(series_id, historical_sales, origin, horizon, feature_names):
        requested = list(feature_names)
        if requested != FEATURE_NAMES:
            raise ToolError("Requested LightGBM feature schema/order is incompatible")
        if not isinstance(origin, int) or isinstance(origin, bool) or origin < 56:
            raise ToolError("LightGBM origin must be an integer with at least 56 historical days")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon != 28:
            raise ToolError("Exported LightGBM snapshots require a 28-day horizon")
        history = np.asarray(historical_sales, dtype=float)
        if history.ndim != 1 or len(history) != origin:
            raise ToolError("Historical sales must contain exactly d_1 through the forecast origin")
        if not np.isfinite(history).all() or (history < 0).any():
            raise ToolError("Historical sales must be finite and nonnegative")
        mapping = _load_mapping(mapping_file)
        canonical_id = normalize_series_id(series_id)
        code = mapping["canonical_series_id_to_code"].get(canonical_id)
        if code is None:
            raise ToolError(
                "Unknown series_id for trained LightGBM mapping: "
                f"{series_id!r} (canonical identity {canonical_id!r})"
            )
        snapshot = historical_snapshot(history.reshape(1, -1))
        rows = {name: np.repeat(value, horizon) for name, value in snapshot.items()}
        rows["series_code"] = np.repeat(code, horizon)
        rows["horizon"] = np.arange(1, horizon + 1, dtype=np.int16)
        dates = _calendar_dates(calendar_file, origin, horizon)
        rows["day_of_week"] = dates.dayofweek.to_numpy(dtype=np.int8)
        rows["month"] = dates.month.to_numpy(dtype=np.int8)
        frame = pd.DataFrame(rows)[FEATURE_NAMES]
        frame["series_code"] = pd.Categorical(
            frame["series_code"], categories=list(range(len(mapping["training_series_ids"])))
        )
        return frame

    return provider


feature_provider = make_feature_provider()
