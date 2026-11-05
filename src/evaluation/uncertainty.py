"""Cluster-aware paired model uncertainty summaries."""
from __future__ import annotations
import numpy as np


def paired_bootstrap(delta_by_cluster, *, n_bootstrap=2000, confidence=0.95, seed=42):
    delta=np.asarray(delta_by_cluster,float)
    delta=delta[np.isfinite(delta)]
    if delta.size < 2: raise ValueError("at least two finite clusters are required")
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,delta.size,replace=True).mean() for _ in range(n_bootstrap)])
    tail=(1-confidence)/2
    return {"clusters":int(delta.size),"mean_difference":float(delta.mean()),
            "median_difference":float(np.median(delta)),"percent_improved":float(100*np.mean(delta<0)),
            "ci_lower":float(np.quantile(means,tail)),"ci_upper":float(np.quantile(means,1-tail))}

def paired_model_comparison(records,model_a,model_b,*,cluster="item_id",n_bootstrap=2000,seed=42):
    """Compare per-series/origin MAE, resampling product clusters rather than days."""
    subset=records[records.model.isin([model_a,model_b])].copy()
    keys=["scenario","series_id","forecast_origin"]
    errors=(subset.assign(abs_error=lambda x:(x.prediction-x.actual).abs())
            .groupby(keys+["model"],as_index=False).abs_error.mean()
            .pivot(index=keys,columns="model",values="abs_error"))
    if model_a not in errors or model_b not in errors or errors[[model_a,model_b]].isna().any().any():
        raise ValueError("paired comparison requires both models on every matched series/origin")
    errors["delta"]=errors[model_a]-errors[model_b]
    metadata=subset.drop_duplicates("series_id").set_index("series_id")
    errors=errors.reset_index();errors[cluster]=errors.series_id.map(metadata[cluster])
    cluster_delta=errors.groupby(cluster).delta.mean().to_numpy()
    result=paired_bootstrap(cluster_delta,n_bootstrap=n_bootstrap,seed=seed)
    result.update({"model_a":model_a,"model_b":model_b,"delta_definition":f"MAE({model_a}) - MAE({model_b})"})
    return result


def paired_hierarchical_comparison(records, model_a, model_b, *, n_bootstrap=2000,
                                   confidence=.95, seed=42):
    """Two-stage bootstrap over origins then product clusters.

    Forecast days are first reduced to series-origin MAE. Each replicate samples
    origins (capturing shared temporal shocks), then product clusters within each
    sampled origin (capturing products sold across stores). With six origins the
    temporal interval remains approximate and should not be called definitive.
    """
    subset=records[records.model.isin([model_a,model_b])].copy()
    keys=["scenario","series_id","forecast_origin"]
    errors=(subset.assign(abs_error=lambda x:(x.prediction-x.actual).abs())
            .groupby(keys+["model"],as_index=False).abs_error.mean()
            .pivot(index=keys,columns="model",values="abs_error"))
    if model_a not in errors or model_b not in errors or errors[[model_a,model_b]].isna().any().any():
        raise ValueError("hierarchical comparison requires both models on every series/origin")
    delta=errors.reset_index();delta["delta"]=delta[model_a]-delta[model_b]
    metadata=subset[["series_id","item_id"]].drop_duplicates("series_id").set_index("series_id")
    delta["item_id"]=delta.series_id.map(metadata.item_id)
    origins=sorted(delta.forecast_origin.unique())
    if len(origins)<2: raise ValueError("at least two forecast origins are required")
    rng=np.random.default_rng(seed);means=[]
    for _ in range(n_bootstrap):
        sampled_origins=rng.choice(origins,len(origins),replace=True);parts=[]
        for origin in sampled_origins:
            frame=delta[delta.forecast_origin==origin]
            products=frame.item_id.unique();sampled=rng.choice(products,len(products),replace=True)
            parts.extend(frame.set_index("item_id").loc[sampled].delta.to_numpy())
        means.append(np.mean(parts))
    tail=(1-confidence)/2;means=np.asarray(means)
    result={"model_a":model_a,"model_b":model_b,"mean_difference":float(delta.delta.mean()),
            "median_difference":float(delta.delta.median()),
            "percent_first_model_wins":float(100*(delta.delta<0).mean()),
            "ci_lower":float(np.quantile(means,tail)),"ci_upper":float(np.quantile(means,1-tail)),
            "matched_series":int(delta.series_id.nunique()),"matched_products":int(delta.item_id.nunique()),
            "matched_origins":len(origins),"matched_observations":int(len(subset)//2),
            "bootstrap":"origins then product clusters","seed":seed}
    result["supports_nonzero_difference"]=bool(result["ci_lower"]>0 or result["ci_upper"]<0)
    return result
