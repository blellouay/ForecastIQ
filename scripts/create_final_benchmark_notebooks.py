"""Create the corrected DL and unified benchmark notebooks from reviewed source cells."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1];NB=ROOT/"notebooks"

def write(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python (ForecastIQ)","language":"python","name":"forecastiq-venv"},"language_info":{"name":"python","version":"3.11"}})
    nbf.write(nb,NB/name);print("wrote",name)

md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell

write("03D_nhits_deepar.ipynb",[
md("""# ForecastIQ — Corrected N-HiTS and DeepAR benchmark

This notebook preserves the earlier DL experiment as an exploratory baseline in the timestamped notebook backup. The corrected workflow uses the fixed 400-series evaluation panel, validation-only configuration selection, the same `d_1886–d_1913` explored benchmark, explicit predictive-mean output, and standardized prediction records.

Set `MODE='fast'` for a resource-conscious smoke test or `MODE='research'` for 750-step, three-origin validation. A fast run is **not** full scientific validation. The research settings are deliberately memory-conscious for a 4-GB GPU."""),
code("""from pathlib import Path
import sys,json,time,platform
import numpy as np,pandas as pd,torch,neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS,DeepAR
from neuralforecast.losses.pytorch import DistributionLoss
PROJECT=Path.cwd() if (Path.cwd()/'src').exists() else Path.cwd().parent
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from src.evaluation.evaluation_panel import validate_panel,prediction_records
from src.experiments.config import ExperimentConfig,save_manifest,sha256
from src.evaluation.nested_selection import (ValidationCache,eligible_inner_origins,
    select_configurations,stable_config_id)
from src.forecasting.neural_models import (cuda_diagnostics,neural_candidates,
    save_training_metadata,training_mode)

OUT=PROJECT/'data'/'processed';CONFIG=ExperimentConfig();SEED=42;H=28
panel=pd.read_parquet(OUT/'evaluation_panel.parquet');validate_panel(panel)
days=[f'd_{i}' for i in range(1,1914)]
calendar=pd.read_csv(PROJECT/'data'/'raw'/'calendar.csv',usecols=['d','date']).drop_duplicates('d').set_index('d')
dates=pd.DatetimeIndex(pd.to_datetime(calendar.reindex(days).date));assert not dates.isna().any()
diagnostics=cuda_diagnostics();display(pd.Series(diagnostics,name='value'))
print('NeuralForecast:',neuralforecast.__version__)
DEVICE='gpu' if diagnostics['cuda_available'] else 'cpu';print('Selected device:',DEVICE)
if not diagnostics['cuda_available'] and diagnostics['torch_cuda_build'] is None:
    print('Cause: this interpreter has a CPU-only PyTorch build. Do not reinstall inside this notebook.')"""),
md("""## Validation-only search

Both families use identical chronological origins. DeepAR compares Student-t and Negative Binomial likelihoods on validation only. NeuralForecast's alias column is the Monte-Carlo predictive **mean**; `-median` is retained separately for diagnostics. Negative Binomial is tested, not presumed superior."""),
code("""MODE='fast'  # change to 'research' for meaningful training
RUN_MULTI_ORIGIN=False  # research-only; expensive, and never implied by a single-origin run
mode=training_mode(MODE)
MAX_STEPS=mode.max_steps;VALIDATION_ORIGINS=list(mode.validation_origins)
VAL_CHECK_STEPS=mode.val_check_steps;BATCH_SIZE=mode.batch_size
INPUT_SIZE=mode.input_size;LEARNING_RATE=mode.learning_rate
candidates=neural_candidates(MODE)
for spec in candidates:spec['config_id']=stable_config_id(spec)
CACHE=ValidationCache(PROJECT/'results'/'validation_cache'/'neural')
PANEL_HASH=sha256(OUT/'evaluation_panel.parquet')

def long_frame(origin):
    ids=panel.series_id.astype(str).to_numpy();Y=panel[days[:origin]].to_numpy(np.float32)
    return pd.DataFrame({'unique_id':np.repeat(ids,origin),'ds':np.tile(dates[:origin],len(ids)),'y':Y.reshape(-1)})

def make_model(spec):
    common=dict(h=H,input_size=INPUT_SIZE,max_steps=MAX_STEPS,learning_rate=spec.get('learning_rate',LEARNING_RATE),
                batch_size=BATCH_SIZE,windows_batch_size=mode.windows_batch_size,random_seed=SEED,scaler_type=spec['scaler'],alias=spec['name'],
                early_stop_patience_steps=mode.early_stop_patience_steps,val_check_steps=VAL_CHECK_STEPS,
                accelerator=DEVICE,devices=1,enable_checkpointing=True,default_root_dir=str(PROJECT/'results'/'checkpoints'/spec['name']))
    if spec['family']=='NHITS':
        if spec.get('mlp_units') is not None:common['mlp_units']=spec['mlp_units']
        return NHITS(**common)
    loss=DistributionLoss(distribution=spec['distribution'],num_samples=mode.num_samples)
    return DeepAR(**common,lstm_n_layers=1,lstm_hidden_size=spec['hidden_size'],lstm_dropout=0.,trajectory_samples=100,loss=loss)

def validate_candidates(inner_origins,outer_origin):
    rows=[]
    for spec in candidates:
        for inner in inner_origins:
            key={'panel_hash':PANEL_HASH,'mode':MODE,'outer_origin':outer_origin,'inner_origin':inner,
                 'config_id':spec['config_id'],'seed':SEED,'max_steps':MAX_STEPS}
            cached=CACHE.load(key)
            if cached is not None:rows.append(cached['result']);continue
            start=time.time()
            base={**spec,'outer_origin':outer_origin,'inner_origin':inner,'seed':SEED,
                  'training_cutoff':inner,'evaluation_population':len(panel)}
            try:
                nf=NeuralForecast(models=[make_model(spec)],freq='D');nf.fit(df=long_frame(inner),val_size=H)
                pred=nf.predict().reset_index();actual=panel[[f'd_{i}' for i in range(inner+1,inner+H+1)]].to_numpy(float).reshape(-1)
                point=pred[spec['name']].to_numpy(float)
                if point.shape!=actual.shape or not np.isfinite(point).all():raise ValueError('invalid forecast shape or nonfinite prediction')
                result={**base,'MAE':float(np.abs(point-actual).mean()),'bias':float((point-actual).mean()),
                        'WAPE':float(np.abs(point-actual).sum()/actual.sum()) if actual.sum()>0 else np.nan,
                        'negative_fraction':float((point<0).mean()),'prediction_mean':float(point.mean()),
                        'actual_mean':float(actual.mean()),'point_forecast':'predictive_sample_mean' if spec['family']=='DeepAR' else 'point',
                        'seconds':time.time()-start,'status':'completed','device':DEVICE}
            except Exception as error:
                result={**base,'MAE':np.nan,'bias':np.nan,'seconds':time.time()-start,
                        'status':'failed','error':repr(error),'device':DEVICE}
            CACHE.save(key,result);rows.append(result)
    return pd.DataFrame(rows)

validation=validate_candidates(VALIDATION_ORIGINS,1885)
selected_summary,validation_leaderboard=select_configurations(validation,expected_origins=VALIDATION_ORIGINS)
display(validation_leaderboard);display(selected_summary)
validation.to_csv(OUT/'phase3d_validation_search.csv',index=False)
validation_leaderboard.to_csv(OUT/'phase3d_validation_leaderboard.csv',index=False)
selected_summary.to_csv(OUT/'phase3d_selected_configurations.csv',index=False)"""),
md("""## Refit validation winners and forecast the explored final period

The final period has been inspected previously and is not described as pristine. Configuration choices above use validation only."""),
code("""selected=selected_summary.to_dict('records');all_records=[];training=[]
for chosen in selected:
    spec=next(c for c in candidates if c['config_id']==chosen['config_id']);start=time.time()
    nf=NeuralForecast(models=[make_model(spec)],freq='D');nf.fit(df=long_frame(1885),val_size=28)
    pred=nf.predict().reset_index();point=pred[spec['name']].to_numpy(float).reshape(len(panel),H)
    records=prediction_records(panel,point,spec['family'],'seen','phase3d-corrected',1885)
    records['point_forecast']='predictive_sample_mean' if spec['family']=='DeepAR' else 'point'
    records['configuration']=spec['name'];all_records.append(records)
    elapsed=time.time()-start
    checkpoint_dir=PROJECT/'results'/'checkpoints'/spec['name']
    checkpoints=[str(p.relative_to(PROJECT)) for p in checkpoint_dir.rglob('*.ckpt')]
    outcome={'model':spec['family'],'configuration':spec['name'],'seconds':elapsed,'device':DEVICE,
             'steps_requested':MAX_STEPS,'checkpoint_files':checkpoints,
             'training_loss':'recorded by Lightning when logger is configured; not fabricated here',
             'early_stopping':'enabled; inspect checkpoint callback/logs'}
    training.append(outcome)
    save_training_metadata(PROJECT/'results'/'training_diagnostics'/f"{spec['name']}_{MODE}.json",
                           mode=mode,candidate=spec,diagnostics=diagnostics,outcome=outcome)
standardized=pd.concat(all_records,ignore_index=True);standardized.to_csv(OUT/'phase3d_standardized_predictions.csv',index=False)
pd.DataFrame(training).to_csv(OUT/'phase3d_training_times.csv',index=False)
diagnostic=(standardized.assign(error=lambda x:x.prediction-x.actual,absolute_error=lambda x:(x.prediction-x.actual).abs())
            .groupby(['model','demand_type'],as_index=False).agg(MAE=('absolute_error','mean'),bias=('error','mean')))
diagnostic.to_csv(OUT/'phase3d_demand_type_diagnostics.csv',index=False);display(diagnostic)
manifest=save_manifest(PROJECT/'results'/'manifests'/'phase3d_corrected.json','phase3d-corrected',CONFIG,
    sources=[OUT/'evaluation_panel.parquet'],extra={'mode':MODE,'device':DEVICE,'candidates':candidates,'validation_origins':VALIDATION_ORIGINS,
    'run_multi_origin':RUN_MULTI_ORIGIN,'selection_metric':'mean validation MAE'})
print('Saved',len(standardized),'matched prediction rows');print('Manifest:',manifest['experiment_id'])"""),
md("""## Optional matched deep-learning multi-origin backtest

This is compute-gated. Every outer origin runs its own inner validation using only target windows ending by that outer origin. Cached results are keyed by panel, configuration, seed, mode, and both origins. It writes selection history and a separate prediction artifact; partial work is never upgraded to a complete benchmark."""),
code("""if RUN_MULTI_ORIGIN:
    if MODE!='research':raise ValueError('Multi-origin scientific results require MODE=research')
    multi=[];nested_validation=[];selection_history=[]
    for outer in CONFIG.backtest_origins:
        inner=eligible_inner_origins(outer,horizon=H,n_folds=3)
        outer_validation=validate_candidates(inner,outer);nested_validation.append(outer_validation)
        outer_selected,outer_board=select_configurations(outer_validation,expected_origins=inner)
        for row in outer_selected.to_dict('records'):
            spec=next(c for c in candidates if c['config_id']==row['config_id'])
            selection_history.append({**row,'outer_origin':outer,'inner_origins':list(inner),
                'training_cutoff':outer,'seed':SEED,'evaluation_population':len(panel)})
            start=time.time();nf=NeuralForecast(models=[make_model(spec)],freq='D');nf.fit(df=long_frame(outer),val_size=H)
            raw=nf.predict().reset_index()[spec['name']].to_numpy(float).reshape(len(panel),H)
            multi.append(prediction_records(panel,raw,spec['family'],'seen','dl-research-nested',outer))
    dl_multi=pd.concat(multi,ignore_index=True)
    dl_multi.to_csv(OUT/'phase3d_multi_origin_predictions.csv',index=False)
    pd.concat(nested_validation,ignore_index=True).to_csv(OUT/'phase3d_nested_validation_history.csv',index=False)
    pd.DataFrame(selection_history).to_csv(OUT/'phase3d_nested_selection_history.csv',index=False)
    print('Saved',len(dl_multi),'DL multi-origin rows')
else:
    print('DL multi-origin not executed; phase3d_multi_origin_predictions.csv was not modified.')""")
])

write("04_unified_forecasting_benchmark.ipynb",[
md("""# ForecastIQ — Unified matched forecasting benchmark

This notebook constructs a matched Scenario-A leaderboard for all available models on the fixed 400-series panel and `d_1886–d_1913`. It reports unavailable or misaligned models explicitly. Scenario B is kept separate and is not ranked with Scenario A. The final period is an **explored benchmark**; multi-origin results are required for stronger claims."""),
code("""from pathlib import Path
import sys,numpy as np,pandas as pd,matplotlib.pyplot as plt
PROJECT=Path.cwd() if (Path.cwd()/'src').exists() else Path.cwd().parent
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from src.evaluation.evaluation_panel import validate_panel,prediction_records
from src.evaluation.benchmark import tune_statistical_panel,leaderboard_from_records,multi_origin_statistical
from src.evaluation.comparison import assert_matched_predictions
from src.evaluation.uncertainty import paired_model_comparison,paired_hierarchical_comparison
from src.evaluation.nested_selection import assert_nested_history
from src.evaluation.multi_origin import (assert_complete_matched_backtest,coverage_table,
    diagnostic_breakdowns,multi_origin_lightgbm)
from src.experiments.config import ExperimentConfig,save_manifest
OUT=PROJECT/'data'/'processed';CONFIG=ExperimentConfig()
panel=pd.read_parquet(OUT/'evaluation_panel.parquet');validate_panel(panel);ids=set(panel.series_id)
status=[]"""),
md("## Statistical models — identical panel, validation-only alpha selection"),
code("""stat_records,stat_validation,stat_selection=tune_statistical_panel(panel,validation_origins=CONFIG.validation_origins,
    final_origin=CONFIG.test_origin,horizon=CONFIG.horizon,alphas=CONFIG.alphas,experiment_id='unified-explored-final')
stat_validation.to_csv(OUT/'unified_statistical_validation.csv',index=False)
stat_selection.to_csv(OUT/'unified_statistical_selection.csv',index=False)
status.extend({'model':m,'scenario':'seen','status':'available'} for m in sorted(stat_records.model.unique()))"""),
md("## Normalize existing ML and corrected DL predictions"),
code("""def normalize_ml(path,source_model,target_model):
    raw=pd.read_csv(path);raw=raw[raw.model==source_model].copy()
    if set(raw.series_id)!=ids:raise ValueError(f'{target_model} series do not match evaluation panel')
    raw=raw.merge(panel[['series_id','item_id','store_id','demand_type']],on=['series_id','demand_type'],validate='many_to_one')
    raw['experiment_id']='unified-explored-final';raw['scenario']='seen';raw['model']=target_model
    raw['forecast_origin']='d_1885';raw['target_day']='d_'+(1885+raw.horizon.astype(int)).astype(str);raw['horizon_step']=raw.horizon.astype(int)
    return raw[['experiment_id','scenario','model','series_id','item_id','store_id','demand_type','forecast_origin','target_day','horizon_step','actual','prediction']]

records=[stat_records]
sources=[(OUT/'phase3a_all_test_forecasts.csv','LightGBM_B_calendar','Direct LightGBM'),
         (OUT/'phase3b_all_test_forecasts.csv','Hurdle_LightGBM','Hurdle LightGBM')]
for path,source,target in sources:
    try:records.append(normalize_ml(path,source,target));status.append({'model':target,'scenario':'seen','status':'available'})
    except Exception as error:status.append({'model':target,'scenario':'seen','status':f'failed: {error}'})
dl_path=OUT/'phase3d_standardized_predictions.csv'
if dl_path.exists():
    dl=pd.read_csv(dl_path)
    for model in ['NHITS','DeepAR']:
        part=dl[dl.model==model]
        if set(part.series_id)==ids:records.append(part[stat_records.columns]);status.append({'model':model,'scenario':'seen','status':'available'})
        else:status.append({'model':model,'scenario':'seen','status':'failed: panel mismatch'})
else:
    status.extend([{'model':'NHITS','scenario':'seen','status':'not run: execute corrected Notebook 3D'},
                   {'model':'DeepAR','scenario':'seen','status':'not run: execute corrected Notebook 3D'}])
status=pd.DataFrame(status);display(status)"""),
md("## Strict matched leaderboard, diagnostics, and paired uncertainty"),
code("""available=pd.concat(records,ignore_index=True);assert_matched_predictions(available)
leaderboard,series_metrics=leaderboard_from_records(available,panel)
leaderboard.to_csv(OUT/'unified_explored_final_leaderboard.csv',index=False);available.to_csv(OUT/'unified_standardized_predictions.csv',index=False)
display(leaderboard.round(4));leaderboard.plot.bar(x='model',y='MAE',legend=False,title='Matched MAE — explored final period');plt.tight_layout();plt.show()
breakdowns=diagnostic_breakdowns(available,panel)
for name,table in breakdowns.items():
    table.to_csv(OUT/f'unified_explored_breakdown_{name}.csv',index=False)
display(breakdowns['demand_type'].pivot(index='demand_type',columns='model',values='MAE').round(3))
pairs=[('TSB','Direct LightGBM'),('TSB','Hurdle LightGBM'),('Direct LightGBM','Hurdle LightGBM'),
       ('Direct LightGBM','NHITS'),('Direct LightGBM','DeepAR'),('NHITS','DeepAR')]
comparisons=[]
for a,b in pairs:
    if {a,b}.issubset(set(available.model)):
        comparisons.append(paired_model_comparison(available,a,b,cluster='item_id'))
pd.DataFrame(comparisons).to_csv(OUT/'unified_paired_comparisons.csv',index=False);display(pd.DataFrame(comparisons))
print('Scenario B is not ranked here: no standardized unseen-series predictions are currently available.')"""),
md("""## Multi-origin protocol

The predeclared non-overlapping origins are read from `ExperimentConfig.backtest_origins`. Statistical models run immediately. Set `RUN_ML_MULTI_ORIGIN=True` for the feasible LightGBM refits. DL records are loaded only from Notebook 3D's explicit research run. Coverage is checked for every model/origin; incomplete work remains visible and is excluded from the complete leaderboard."""),
code("""RUN_ML_MULTI_ORIGIN=False
multi_records,multi_validation,stat_nested_selection=multi_origin_statistical(panel,CONFIG.backtest_origins,horizon=CONFIG.horizon,
    alphas=CONFIG.alphas,experiment_id='statistical-multi-origin',return_selection=True)
multi_records.to_csv(OUT/'unified_statistical_multi_origin_predictions.csv',index=False)
multi_validation.to_csv(OUT/'unified_statistical_multi_origin_validation.csv',index=False)
stat_nested_selection.to_csv(OUT/'unified_statistical_nested_selection.csv',index=False)
stat_boundaries=(multi_validation[['evaluation_origin','inner_validation_origins']].drop_duplicates('evaluation_origin')
                 .explode('inner_validation_origins').rename(columns={'evaluation_origin':'outer_origin','inner_validation_origins':'inner_origin'}))
assert_nested_history(stat_boundaries,horizon=CONFIG.horizon)
if RUN_ML_MULTI_ORIGIN:
    ml_multi,ml_history=multi_origin_lightgbm(panel,CONFIG.backtest_origins,horizon=CONFIG.horizon,
                                   experiment_id='lightgbm-multi-origin',return_history=True)
    ml_multi.to_csv(OUT/'unified_ml_multi_origin_predictions.csv',index=False)
    ml_history.to_csv(OUT/'unified_ml_selection_history.csv',index=False)
ml_path=OUT/'unified_ml_multi_origin_predictions.csv'
dl_multi_path=OUT/'phase3d_multi_origin_predictions.csv'
multi_parts=[multi_records]
if ml_path.exists():multi_parts.append(pd.read_csv(ml_path))
if dl_multi_path.exists():multi_parts.append(pd.read_csv(dl_multi_path))
multi_available=pd.concat(multi_parts,ignore_index=True)
all_models=['Average','Naive','SeasonalNaive','SES','Croston','SBA','TSB',
            'Direct LightGBM','Hurdle LightGBM','NHITS','DeepAR']
coverage=coverage_table(multi_available,all_models,CONFIG.backtest_origins,
                        series_count=len(panel),horizon=CONFIG.horizon)
coverage.to_csv(OUT/'unified_multi_origin_coverage.csv',index=False);display(coverage)
complete_models=coverage.groupby('model').status.apply(lambda s:(s=='complete').all())
complete_models=complete_models[complete_models].index.tolist()
complete=multi_available[multi_available.model.isin(complete_models)].copy()
if complete_models:
    assert_complete_matched_backtest(complete,complete_models,CONFIG.backtest_origins,
                                     series_count=len(panel),horizon=CONFIG.horizon)
multi_summary=(complete.assign(abs_error=lambda x:(x.prediction-x.actual).abs())
               .groupby(['model','forecast_origin'],as_index=False).abs_error.mean())
display(multi_summary.pivot(index='model',columns='forecast_origin',values='abs_error').round(3))
if len(complete):
    nested_breakdowns=diagnostic_breakdowns(complete,panel)
    for name,table in nested_breakdowns.items():table.to_csv(OUT/f'unified_nested_breakdown_{name}.csv',index=False)
    multi_summary.pivot(index='forecast_origin',columns='model',values='abs_error').plot(marker='o',title='Nested MAE by origin')
    plt.tight_layout();plt.show()
multi_pairs=[]
for a,b in pairs:
    if {a,b}.issubset(complete_models):
        multi_pairs.append(paired_hierarchical_comparison(complete,a,b,n_bootstrap=2000,seed=42))
pd.DataFrame(multi_pairs).to_csv(OUT/'unified_multi_origin_paired_comparisons.csv',index=False)
display(pd.DataFrame(multi_pairs))
print('Complete models:',complete_models)
print('Incomplete models are shown in coverage and excluded, never silently mixed.')"""),
code("""manifest=save_manifest(PROJECT/'results'/'manifests'/'unified_benchmark.json','unified-benchmark',CONFIG,
    sources=[OUT/'evaluation_panel.parquet'],extra={'available_models':sorted(available.model.unique()),
    'failed_or_missing':status[status.status!='available'].to_dict('records'),'final_period_status':'previously explored',
    'multi_origin_complete_models':complete_models,'multi_origin_coverage':coverage.to_dict('records')})
print('Manifest:',manifest['experiment_id'])""")
])
