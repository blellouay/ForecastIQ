"""Matched multi-origin protocol utilities shared by notebooks and agents."""
from __future__ import annotations

from collections.abc import Iterable
import pandas as pd
import numpy as np

REQUIRED_KEYS = ("series_id", "forecast_origin", "horizon_step", "target_day")


def _canonical(records: pd.DataFrame) -> pd.DataFrame:
    """Accept legacy generic horizon/date names, emit the project schema."""
    rename = {}
    if "horizon_step" not in records and "horizon" in records:
        rename["horizon"] = "horizon_step"
    if "target_day" not in records and "target_date" in records:
        rename["target_date"] = "target_day"
    return records.rename(columns=rename)


def _origin_mask(series: pd.Series, origin: int) -> pd.Series:
    text = series.astype(str).str.replace(r"^d_", "", regex=True)
    return pd.to_numeric(text, errors="coerce") == int(origin)


def coverage_table(records: pd.DataFrame, models: Iterable[str],
                   origins: Iterable[int], *, series_count: int,
                   horizon: int) -> pd.DataFrame:
    """Report expected/observed rows; partial runs can never look complete."""
    expected = series_count * horizon
    records = _canonical(records)
    rows = []
    for model in models:
        for origin in origins:
            part = records[(records.model == model) & _origin_mask(records.forecast_origin, origin)]
            unique = part.drop_duplicates(list(REQUIRED_KEYS)).shape[0] if len(part) else 0
            rows.append({"model": model, "forecast_origin": int(origin),
                         "rows": int(len(part)), "unique_rows": int(unique),
                         "expected_rows": expected,
                         "status": "complete" if len(part) == unique == expected else "incomplete"})
    return pd.DataFrame(rows)


def assert_complete_matched_backtest(records: pd.DataFrame, models: Iterable[str],
                                     origins: Iterable[int], *, series_count: int,
                                     horizon: int) -> None:
    records = _canonical(records)
    missing = set((*REQUIRED_KEYS, "model", "actual", "prediction")) - set(records)
    if missing:
        raise ValueError(f"prediction records missing columns: {sorted(missing)}")
    coverage = coverage_table(records, models, origins, series_count=series_count,
                              horizon=horizon)
    bad = coverage[coverage.status != "complete"]
    if len(bad):
        details = bad[["model", "forecast_origin", "rows", "expected_rows"]].to_dict("records")
        raise ValueError(f"incomplete multi-origin benchmark: {details}")

    reference = None
    for model in models:
        keys = set(map(tuple, records[records.model == model][list(REQUIRED_KEYS)].to_numpy()))
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError(f"{model} does not forecast the same keys as other models")


def diagnostic_breakdowns(records: pd.DataFrame, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return comparable MAE/bias tables across requested diagnostic dimensions."""
    missing_meta = [c for c in ("store_id", "demand_type", "average_demand")
                    if c in panel and c not in records]
    meta_cols = ["series_id", *missing_meta]
    frame = records.merge(panel[meta_cols].drop_duplicates("series_id"),
                          on="series_id", how="left") if missing_meta else records.copy()
    frame["error"] = frame.prediction - frame.actual
    frame["absolute_error"] = frame.error.abs()
    horizon_column = "horizon_step" if "horizon_step" in frame else "horizon"
    frame["horizon_bucket"] = pd.cut(frame[horizon_column], [0, 7, 14, 21, 28], include_lowest=True)
    if "average_demand" in frame:
        frame["volume_bucket"] = pd.qcut(frame.average_demand.rank(method="first"), 4,
                                          labels=["low", "medium", "high", "very_high"])

    result = {}
    for name, columns in {
        "overall": [], "demand_type": ["demand_type"], "store": ["store_id"],
        "volume": ["volume_bucket"], "horizon": ["horizon_bucket"],
        "origin": ["forecast_origin"],
    }.items():
        usable = [c for c in columns if c in frame]
        grouped = frame.groupby(["model", *usable], observed=True, dropna=False)
        result[name] = grouped.agg(MAE=("absolute_error", "mean"),
                                   bias=("error", "mean"), rows=("error", "size")).reset_index()
    return result


def rolling_training_origins(origin: int, *, horizon: int = 28,
                             folds: int = 6) -> tuple[int, ...]:
    """Predeclare non-overlapping training targets strictly before an evaluation origin."""
    first = origin - folds * horizon
    if first < 56:
        raise ValueError("insufficient history for requested training folds")
    return tuple(range(first, origin, horizon))


def multi_origin_lightgbm(panel: pd.DataFrame, origins: Iterable[int], *,
                          horizon: int = 28, dates=None, params=None,
                          experiment_id: str = "lightgbm-multi-origin",
                          return_history: bool = False):
    """Fit Direct and Hurdle LightGBM on identical origin-safe rows.

    The panel must contain metadata plus ``d_1...`` demand columns.  Each model at
    each evaluation origin is trained on earlier, non-overlapping 28-day targets.
    """
    from src.evaluation.evaluation_panel import prediction_records
    from src.features.feature_engineering import fixed_origin_frame
    from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames

    day_cols = sorted((c for c in panel if c.startswith("d_")), key=lambda x: int(x[2:]))
    demand = panel[day_cols].to_numpy(np.float32)
    output = [];history=[]
    for origin in origins:
        train_origins = rolling_training_origins(int(origin), horizon=horizon)
        train = pd.concat([fixed_origin_frame(demand, o, horizon, dates=dates,
                                               include_target=True)
                           for o in train_origins], ignore_index=True)
        future = fixed_origin_frame(demand, int(origin), horizon, dates=dates)
        y = train.pop("target")
        fitted = fit_predict_hurdle_and_direct_frames(train, y, future, params=params)
        for key, model in (("direct", "Direct LightGBM"), ("hurdle", "Hurdle LightGBM")):
            prediction = fitted[key].reshape(len(panel), horizon)
            output.append(prediction_records(panel, prediction, model, "seen",
                                             experiment_id, int(origin)))
            history.append({"model":model,"forecast_origin":int(origin),
                            "inner_training_origins":list(train_origins),
                            "training_cutoff":int(origin),"candidate_configurations":1,
                            "selected_configuration":params or "predeclared defaults",
                            "selection_criterion":"predeclared fixed configuration; no outer-data tuning",
                            "random_seed":(params or {}).get("random_state",42),
                            "training_rows":int(len(train)),"successful_folds":len(train_origins)})
    records=pd.concat(output, ignore_index=True)
    return (records,pd.DataFrame(history)) if return_history else records
