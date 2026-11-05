"""Small deterministic Scenario-B and simulated cold-start evaluation."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.agents.generalizable_routing import predict_generalizable_snapshot
from src.forecasting.statistical_models import MODELS,forecast
from src.forecasting.cold_start import comparable_product_forecast
from src.evaluation.metrics import mae,mase,rmsse,wape,mean_signed_bias

def main(n_unseen=20,n_cold=5):
    raw=pd.read_csv(ROOT/'data/raw/sales_train_validation.csv')
    mapping=json.loads((ROOT/'models/lightgbm/series_mapping.json').read_text())['series_id_to_code']
    canonical=raw.id.str.replace('_validation','',regex=False).str.replace(r'_((?:CA|TX|WI)_\d+)$',r'__\1',regex=True)
    panel=raw.loc[~canonical.isin(mapping)].head(n_unseen).copy();origins=(1801,1829,1857)
    rows=[];registry=ROOT/'models/generalizable_lightgbm/registry.json'
    models=[*MODELS,'GeneralizableDirectLightGBM','GeneralizableHurdleLightGBM']
    for _,record in panel.iterrows():
        for origin in origins:
            history=np.array([record[f'd_{i}'] for i in range(1,origin+1)],float)
            truth=np.array([record[f'd_{i}'] for i in range(origin+1,origin+29)],float)
            for model in models:
                pred=(forecast(model,history,28) if model in MODELS else
                      np.array(predict_generalizable_snapshot(record.id,history.tolist(),origin,28,model,registry)['forecast']))
                rows.append({'series_id':record.id,'origin':origin,'model':model,'MAE':mae(truth,pred),
                    'MASE':mase(truth,pred,history),'RMSSE':rmsse(truth,pred,history),
                    'WAPE':wape(truth,pred),'bias':mean_signed_bias(truth,pred)})
    detail=pd.DataFrame(rows);out=ROOT/'data/processed';detail.to_csv(out/'hybrid_pilot_unseen_detail.csv',index=False)
    summary=detail.groupby('model',as_index=False).agg(series_folds=('MAE','size'),MAE=('MAE','mean'),
        MASE=('MASE','mean'),RMSSE=('RMSSE','mean'),WAPE=('WAPE','mean'),bias=('bias','mean')).sort_values('MAE')
    summary.to_csv(out/'hybrid_pilot_unseen_summary.csv',index=False)
    cold=[]
    for _,record in panel.head(n_cold).iterrows():
        item=record.item_id;meta={'category':item.split('_')[0],'department':'_'.join(item.split('_')[:2]),'store':record.store_id}
        prediction=comparable_product_forecast(record.id,meta,ROOT/'data/processed/m5_sample_400.csv',1885)
        truth=np.array([record[f'd_{i}'] for i in range(1886,1914)],float);pred=np.array(prediction['forecast'])
        cold.append({'series_id':record.id,'MAE':mae(truth,pred),'actual_total':truth.sum(),
                     'predicted_total':pred.sum(),'comparables':prediction['metadata']['comparable_products']})
    cold_df=pd.DataFrame(cold);cold_df.to_csv(out/'hybrid_pilot_cold_start.csv',index=False)
    manifest={'experiment':'hybrid-generalizable-pilot','scenario_b_series':n_unseen,
        'scenario_b_origins':list(origins),'training_population':'frozen 400-series panel',
        'evaluation_population':'first deterministic M5 identities excluded from the 400-series mapping',
        'item_group_holdout':False,'cold_start_series':n_cold,
        'cold_start_is_simulated':True,'results':summary.to_dict('records'),
        'cold_start_mean_mae':float(cold_df.MAE.mean()),
        'limitations':['Pilot sample only','No genuine launch dates','Item IDs may appear at other stores in training']}
    (ROOT/'results/manifests/hybrid_generalizable_pilot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False));print('cold_start_mean_mae',manifest['cold_start_mean_mae'])

if __name__=='__main__':main()
