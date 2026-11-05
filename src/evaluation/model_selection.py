"""Validation-only model-selection strategies with deterministic ties."""
from __future__ import annotations
import pandas as pd

def _winner(group, minimum_improvement=0.0, baseline=None, expected_origins=None):
    if expected_origins is not None:
        if "validation_origin" not in group:
            raise ValueError("validation_origin is required when expected_origins is provided")
        expected=set(map(int,expected_origins)); complete=[]
        for model,part in group.groupby("model"):
            observed=set(part.validation_origin.astype(int))
            if observed==expected: complete.append(model)
        group=group[group.model.isin(complete)]
        if group.empty: raise ValueError("no model completed every expected validation origin")
    scores=group.groupby("model",as_index=False)["validation_loss"].mean().sort_values(["validation_loss","model"],kind="stable")
    winner=scores.iloc[0]
    if baseline and baseline in set(scores.model):
        base=scores.loc[scores.model==baseline].iloc[0]
        if base.validation_loss-winner.validation_loss < minimum_improvement: return baseline
    return str(winner.model)

def select_models(validation, strategy, *, minimum_improvement=0.0, baseline=None,expected_origins=None):
    required={"series_id","demand_type","model","validation_loss"}
    if not required.issubset(validation): raise ValueError(f"missing columns: {sorted(required-set(validation))}")
    if strategy=="global": return {"__global__":_winner(validation,minimum_improvement,baseline,expected_origins)}
    key="demand_type" if strategy=="per_demand_type" else "series_id" if strategy=="per_series" else None
    if key is None: raise ValueError("unknown selection strategy")
    return {str(name):_winner(group,minimum_improvement,baseline,expected_origins) for name,group in validation.groupby(key,sort=True)}
