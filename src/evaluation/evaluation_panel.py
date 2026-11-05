"""Deterministic, versioned evaluation-panel and prediction-record contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

PANEL_SEED = 42
CATEGORIES = ("Smooth", "Erratic", "Intermittent", "Lumpy")
PREDICTION_COLUMNS = ["experiment_id","scenario","model","series_id","item_id","store_id",
                      "demand_type","forecast_origin","target_day","horizon_step","actual","prediction"]

@dataclass(frozen=True)
class PanelProtocol:
    seed:int=PANEL_SEED
    per_category:int=100
    forecast_origin:int=1885
    horizon:int=28
    validation_origins:tuple=(1801,1829,1857)
    backtest_origins:tuple=(1717,1745,1773,1801,1829,1857)
    category_cutoff:int=1885

def validate_panel(panel, protocol=PanelProtocol()):
    required={"item_id","store_id","series_id","demand_type"}
    missing=required-set(panel)
    if missing: raise ValueError(f"panel missing columns: {sorted(missing)}")
    if panel.series_id.duplicated().any() or panel.duplicated(["item_id","store_id"]).any():
        raise ValueError("evaluation panel contains duplicate product-store series")
    counts=panel.demand_type.value_counts()
    for category in CATEGORIES:
        if counts.get(category,0)!=protocol.per_category:
            raise ValueError(f"expected {protocol.per_category} {category} series")
    days=sorted([c for c in panel if c.startswith("d_") and c[2:].isdigit()],key=lambda c:int(c[2:]))
    if len(days)!=1913 or days[protocol.forecast_origin-1]!="d_1885":
        raise ValueError("panel does not match the 1,913-day M5 accuracy timeline")
    return True

def create_evaluation_panel(full_sales,metadata,output_path,*,protocol=PanelProtocol(),reuse_ids=None):
    sales=pd.read_csv(full_sales);meta=pd.read_csv(metadata)
    if "series_id" not in sales: sales.insert(0,"series_id",sales.item_id.astype(str)+"__"+sales.store_id.astype(str))
    frame=sales.merge(meta.drop(columns=[c for c in ["item_id","store_id"] if c in meta]),on="series_id",validate="one_to_one")
    if reuse_ids is not None:
        ids=pd.read_csv(reuse_ids).series_id.astype(str)
        panel=frame.set_index("series_id").loc[ids].reset_index()
    else:
        parts=[]
        for category in CATEGORIES:
            group=frame[frame.demand_type==category]
            parts.append(group.sample(protocol.per_category,random_state=protocol.seed))
        panel=pd.concat(parts,ignore_index=True)
    validate_panel(panel,protocol)
    panel["panel_seed"]=protocol.seed;panel["category_cutoff_day"]=protocol.category_cutoff
    panel["final_forecast_origin"]=protocol.forecast_origin;panel["final_target_start"]="d_1886";panel["final_target_end"]="d_1913"
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True);panel.to_parquet(output,index=False)
    manifest={"panel_file":output.name,"rows":len(panel),"seed":protocol.seed,"categories":panel.demand_type.value_counts().to_dict(),
              "forecast_origin":protocol.forecast_origin,"horizon":protocol.horizon,
              "validation_origins":list(protocol.validation_origins),"backtest_origins":list(protocol.backtest_origins),
              "scenario_a":"seen product-store series","scenario_b":"unseen product-store combinations; history available at forecast time"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return panel

def prediction_records(panel,predictions,model,scenario,experiment_id,origin=1885,protocol=PanelProtocol()):
    validate_panel(panel,protocol)
    days=[f"d_{i}" for i in range(origin+1,origin+29)]
    actual=panel[days].to_numpy(float);prediction=np.asarray(predictions,float)
    if prediction.shape!=actual.shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite and aligned to panel x 28 days")
    rows=[]
    for i,row in panel.reset_index(drop=True).iterrows():
        for step,day in enumerate(days,1):
            rows.append({"experiment_id":experiment_id,"scenario":scenario,"model":model,
                "series_id":row.series_id,"item_id":row.item_id,"store_id":row.store_id,"demand_type":row.demand_type,
                "forecast_origin":f"d_{origin}","target_day":day,"horizon_step":step,
                "actual":actual[i,step-1],"prediction":max(float(prediction[i,step-1]),0.)})
    return pd.DataFrame(rows,columns=PREDICTION_COLUMNS)

def assert_unseen_isolation(training_ids,panel_ids):
    overlap=set(map(str,training_ids))&set(map(str,panel_ids))
    if overlap: raise ValueError(f"unseen-series training contains {len(overlap)} evaluation series")
    return True
