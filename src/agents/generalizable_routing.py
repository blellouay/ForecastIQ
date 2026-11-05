"""Inference for the separate ID-free generalizable LightGBM artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from src.agent_tools.forecasting_tools import ToolError
from src.features.generalizable_features import FEATURE_NAMES,PIPELINE_ID,feature_frame
from src.forecasting.generalizable_lightgbm import predict_generalizable

NAMES=('GeneralizableDirectLightGBM','GeneralizableHurdleLightGBM')

def predict_generalizable_snapshot(series_id,history,origin,horizon,model,
                                   registry_path='models/generalizable_lightgbm/registry.json'):
    path=Path(registry_path).resolve()
    if not path.is_file():raise ToolError(f'Generalizable LightGBM registry missing: {path}')
    registry=json.loads(path.read_text(encoding='utf-8'))
    if registry.get('feature_pipeline_id')!=PIPELINE_ID or registry.get('feature_names')!=FEATURE_NAMES:
        raise ToolError('Generalizable LightGBM feature schema mismatch')
    entry=registry.get('models',{}).get(model)
    if not entry:raise ToolError(f'{model} is not registered')
    snap=entry.get('snapshots',{}).get(str(origin))
    if not snap:raise ToolError(f'No generalizable snapshot at origin {origin}')
    if snap.get('trained_through_day',origin+1)>origin:raise ToolError('Generalizable snapshot training cutoff is invalid')
    metadata=json.loads((path.parent/snap['metadata_path']).read_text(encoding='utf-8'))
    if metadata.get('feature_names')!=FEATURE_NAMES:raise ToolError('Generalizable metadata schema mismatch')
    dates=pd.date_range('2011-01-29',periods=1913,freq='D')
    frame=feature_frame([series_id],[history],origin,horizon,dates,metadata['categories'])
    values=predict_generalizable(frame,path.parent,origin,entry['kind'])
    return {'series_id':series_id,'model':model,'horizon':horizon,'forecast':values.tolist(),
            'total_expected_demand':float(values.sum()),'metadata':{
                'source':'generalizable_lightgbm_pilot','forecast_origin_day':origin,
                'trained_through_day':snap['trained_through_day'],'feature_pipeline_id':PIPELINE_ID,
                'training_series_count':metadata['training_series_count'],
                'training_population_contains_series':series_id.replace('_validation','').replace('_evaluation','').replace('_CA_','__CA_').replace('_TX_','__TX_').replace('_WI_','__WI_') in metadata['training_series_ids']}}
