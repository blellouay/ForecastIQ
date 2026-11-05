"""Matched-panel benchmark helpers used by Notebook 4 and future agents."""
from __future__ import annotations
import numpy as np
import pandas as pd
import time
from src.forecasting.statistical_models import forecast
from .evaluation_panel import prediction_records
from .metrics import series_metric_records,aggregate_metric_records

STATISTICAL_MODELS=("Average","Naive","SeasonalNaive","SES","Croston","SBA","TSB")

def tune_statistical_panel(panel,*,validation_origins=(1801,1829,1857),final_origin=1885,horizon=28,alphas=(.1,.2,.3),experiment_id="unified"):
    days=[f"d_{i}" for i in range(1,1914)];Y=panel[days].to_numpy(float)
    predictions={};validation=[];selected=[]
    for model in STATISTICAL_MODELS:
        choices=(tuple((a,b) for a in alphas for b in alphas) if model=="TSB" else
                 tuple(alphas) if model in {"SES","Croston","SBA"} else (None,))
        choice_errors=np.zeros((len(panel),len(choices)))
        for j,choice in enumerate(choices):
            for origin in validation_origins:
                kwargs=({} if choice is None else
                        {"alpha":choice[0],"beta":choice[1]} if model=="TSB" else {"alpha":choice})
                pred=np.vstack([forecast(model,row[:origin],horizon,**kwargs) for row in Y])
                choice_errors[:,j]+=np.abs(pred-Y[:,origin:origin+horizon]).mean(1)/len(validation_origins)
            validation.append({"model":model,"alpha":None if choice is None else choice[0] if model=="TSB" else choice,
                               "beta":choice[1] if model=="TSB" else None,
                               "mean_validation_MAE":float(choice_errors[:,j].mean()),
                               "validation_origins":list(validation_origins),"successful_folds":len(validation_origins)})
        best=np.argmin(choice_errors,axis=1);final=np.empty((len(panel),horizon))
        for i,row in enumerate(Y):
            choice=choices[int(best[i])]
            kwargs=({} if choice is None else {"alpha":choice[0],"beta":choice[1]} if model=="TSB" else {"alpha":choice})
            final[i]=forecast(model,row[:final_origin],horizon,**kwargs)
            selected.append({"series_id":panel.iloc[i].series_id,"model":model,
                             "alpha":None if choice is None else choice[0] if model=="TSB" else choice,
                             "beta":choice[1] if model=="TSB" else None,
                             "validation_MAE":choice_errors[i,best[i]],"forecast_origin":final_origin,
                             "inner_validation_origins":list(validation_origins),
                             "selection_criterion":"minimum mean MAE across eligible inner folds"})
        predictions[model]=final
    records=pd.concat([prediction_records(panel,predictions[m],m,"seen",experiment_id,final_origin) for m in STATISTICAL_MODELS],ignore_index=True)
    return records,pd.DataFrame(validation),pd.DataFrame(selected)

def leaderboard_from_records(records,panel):
    origin=int(str(records.forecast_origin.iloc[0]).split("_")[-1]);days=[f"d_{i}" for i in range(1,origin+1)]
    training=panel.set_index("series_id").loc[records.series_id.unique(),days].to_numpy(float)
    first_model=records.model.iloc[0]
    actual=records[records.model==first_model].pivot(index="series_id",columns="horizon_step",values="actual").loc[records.series_id.unique()].to_numpy()
    preds={m:g.pivot(index="series_id",columns="horizon_step",values="prediction").loc[records.series_id.unique()].to_numpy() for m,g in records.groupby("model")}
    ids=records.series_id.unique();types=panel.set_index("series_id").loc[ids].demand_type.to_numpy()
    rows=series_metric_records(actual,preds,training,ids,types)
    board=rows.groupby("model").apply(aggregate_metric_records,include_groups=False).reset_index().sort_values("MAE")
    return board,rows

def multi_origin_statistical(panel,origins,*,horizon=28,alphas=(.1,.2,.3),experiment_id="multi-origin",return_selection=False):
    """Predeclared rolling evaluation; each origin tunes only on three earlier windows."""
    outputs=[];validation=[];selection=[]
    for origin in origins:
        started=time.perf_counter()
        prior=(origin-3*horizon,origin-2*horizon,origin-horizon)
        if min(prior)<112: raise ValueError("origin lacks sufficient prior validation history")
        records,scores,chosen=tune_statistical_panel(panel,validation_origins=prior,final_origin=origin,
            horizon=horizon,alphas=alphas,experiment_id=experiment_id)
        records["forecast_origin"]=f"d_{origin}"
        scores["evaluation_origin"]=origin;scores["inner_validation_origins"]=[list(prior)]*len(scores)
        chosen["evaluation_origin"]=origin
        chosen["random_seed"]="not applicable: deterministic analytical model"
        chosen["training_duration_seconds"]=time.perf_counter()-started
        chosen["checkpoint"]="not applicable"
        chosen["evaluation_population"]=len(panel)
        chosen["successful_folds"]=len(prior)
        chosen["candidate_configurations"]=chosen.model.map(
            {m:int((scores.model==m).sum()) for m in scores.model.unique()})
        validation.append(scores);selection.append(chosen);outputs.append(records)
    result=(pd.concat(outputs,ignore_index=True),pd.concat(validation,ignore_index=True))
    return (*result,pd.concat(selection,ignore_index=True)) if return_selection else result
