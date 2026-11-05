"""Resource-conscious exporter for the separate generalizable LightGBM family."""
from __future__ import annotations
import argparse,json,os,sys,tempfile,time,hashlib,platform
from importlib.metadata import version
from pathlib import Path
import joblib,numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.features.generalizable_features import (FEATURE_NAMES,CATEGORICAL_FEATURES,
    PIPELINE_ID,feature_frame)
from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames

MODES={'smoke':dict(series=80,origins=2,estimators=50),
       'pilot':dict(series=400,origins=8,estimators=200),
       'research':dict(series=None,origins=16,estimators=250)}
ORIGINS=(1801,1829,1857,1885);H=28;SEED=42

def atomic(value,path,kind='json'):
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        if kind=='joblib':joblib.dump(value,tmp)
        else:Path(tmp).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def main(mode='pilot',threads=4,memory_mb=512):
    cfg=MODES[mode];source=ROOT/'data/processed'/('m5_full_sales.csv' if mode=='research' else 'm5_sample_400.csv')
    panel=pd.read_csv(source);days=sorted([c for c in panel if c.startswith('d_')],key=lambda x:int(x[2:]))
    if cfg['series']:panel=panel.iloc[:cfg['series']].copy()
    ids=panel.series_id.astype(str).tolist();Y=panel[days].to_numpy(np.float32)
    cal=pd.read_csv(ROOT/'data/raw/calendar.csv',usecols=['d','date']).drop_duplicates('d').set_index('d')
    dates=pd.DatetimeIndex(pd.to_datetime(cal.reindex(days).date))
    categories={'store':sorted(panel.store_id.astype(str).unique()),
                'department':sorted(panel.item_id.astype(str).str.split('_').str[:2].str.join('_').unique()),
                'category':sorted(panel.item_id.astype(str).str.split('_').str[0].unique()),
                'demand_type':['Smooth','Erratic','Intermittent','Lumpy']}
    out=ROOT/'models/generalizable_lightgbm';snapshots={};params=dict(
        n_estimators=cfg['estimators'],learning_rate=.05,num_leaves=31,min_child_samples=80,
        reg_lambda=1.,random_state=SEED,n_jobs=threads,verbosity=-1)
    for forecast_origin in ORIGINS:
        eligible=list(range(112,forecast_origin-H+1,H))[-cfg['origins']:]
        frames=[];targets=[];started=time.perf_counter()
        for origin in eligible:
            frames.append(feature_frame(ids,Y[:,:origin],origin,H,dates,categories))
            targets.append(Y[:,origin:origin+H].ravel())
        X=pd.concat(frames,ignore_index=True);y=np.concatenate(targets)
        mb=(X.memory_usage(deep=True).sum()+y.nbytes)/1024**2
        if mb>memory_mb:raise MemoryError(f'Feature matrix {mb:.1f} MiB exceeds budget {memory_mb} MiB')
        result=fit_predict_hurdle_and_direct_frames(X,y,X.iloc[:28].copy(),params=params,
                                                     categorical_features=CATEGORICAL_FEATURES)
        paths={'direct':out/f'direct_origin_{forecast_origin}.joblib',
               'classifier':out/f'hurdle_classifier_origin_{forecast_origin}.joblib',
               'regressor':out/f'hurdle_regressor_origin_{forecast_origin}.joblib'}
        atomic(result['direct_model'],paths['direct'],'joblib');atomic(result['classifier'],paths['classifier'],'joblib');atomic(result['quantity_model'],paths['regressor'],'joblib')
        duration=time.perf_counter()-started
        metadata={'forecast_origin_day':forecast_origin,'trained_through_day':max(eligible)+H,
                  'training_origins':eligible,'training_series_ids':ids,'training_series_count':len(ids),
                  'feature_names':FEATURE_NAMES,'categorical_features':CATEGORICAL_FEATURES,
                  'categories':categories,'feature_pipeline_id':PIPELINE_ID,'hyperparameters':params,
                  'random_seed':SEED,'mode':mode,'training_rows':len(X),'estimated_memory_mb':mb,
                  'duration_seconds':duration,'dataset':str(source.relative_to(ROOT)).replace('\\','/'),
                  'package_versions':{n:version(n) for n in ('lightgbm','pandas','numpy','joblib')},
                  'python':platform.python_version(),'scenario_support':['known_series','unseen_series_with_history']}
        atomic(metadata,out/f'origin_{forecast_origin}.metadata.json')
        snapshots[str(forecast_origin)]={'trained_through_day':metadata['trained_through_day'],
            'forecast_origin_day':forecast_origin,'metadata_path':f'origin_{forecast_origin}.metadata.json'}
        print(f'origin={forecast_origin} rows={len(X):,} memory={mb:.1f}MiB seconds={duration:.2f}',flush=True)
    registry={'schema_version':1,'feature_pipeline_id':PIPELINE_ID,'feature_names':FEATURE_NAMES,
              'categorical_features':CATEGORICAL_FEATURES,'categories':categories,'mode':mode,
              'models':{'GeneralizableDirectLightGBM':{'kind':'direct','snapshots':snapshots},
                        'GeneralizableHurdleLightGBM':{'kind':'hurdle','snapshots':snapshots}}}
    atomic(registry,out/'registry.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,default='pilot');p.add_argument('--threads',type=int,default=4);p.add_argument('--memory-mb',type=int,default=512);a=p.parse_args();main(a.mode,a.threads,a.memory_mb)
