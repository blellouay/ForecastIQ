"""One-time, idempotent migration of the six research notebooks to shared code."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=ROOT/"notebooks"
FILES=["01_data_exploration_full_export.ipynb","02_forecasting_4_workflows_10_seeds.ipynb",
       "02B_full_statistical_benchmark.ipynb","03_global_machine_learning_forecasting.ipynb",
       "03B_hurdle_lightgbm.ipynb","03C_large_scale_lightgbm_hurdle.ipynb"]
BOOTSTRAP="""# Shared scientifically validated ForecastIQ framework\nimport sys\nfrom pathlib import Path\n_project = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nif str(_project) not in sys.path:\n    sys.path.insert(0, str(_project))\nfrom src.experiments.config import ExperimentConfig, save_manifest\nEXPERIMENT_CONFIG = ExperimentConfig()\n"""

def code(source):
    return {"cell_type":"code","execution_count":None,"id":"shared-framework","metadata":{},"outputs":[],
            "source":[line+"\n" for line in source.splitlines()]}

def set_cell(nb,index,source):
    nb["cells"][index]["source"]=[line+"\n" for line in source.splitlines()]
    nb["cells"][index]["execution_count"]=None; nb["cells"][index]["outputs"]=[]

for name in FILES:
    path=NOTEBOOKS/name; nb=json.loads(path.read_text(encoding="utf-8"))
    experiment=name.removesuffix(".ipynb")
    bootstrap=BOOTSTRAP+f"""EXPERIMENT_NAME = '{experiment}'\nMANIFEST = save_manifest(_project / 'results' / 'manifests' / (EXPERIMENT_NAME + '.json'), EXPERIMENT_NAME, EXPERIMENT_CONFIG, sources=[_project / 'data' / 'raw' / 'sales_train_validation.csv'])\nprint('Experiment manifest:', MANIFEST['experiment_id'])"""
    existing=next((c for c in nb["cells"] if c.get("id")=="shared-framework"),None)
    if existing is None: nb["cells"].insert(1,code(bootstrap))
    else: existing["source"]=[line+"\n" for line in bootstrap.splitlines()]
    # Indices below account for the inserted shared-framework cell.
    if name=="02_forecasting_4_workflows_10_seeds.ipynb":
        set_cell(nb,15,"""from src.forecasting.statistical_models import (\n    forecast_average, forecast_naive, forecast_seasonal_naive, forecast_ses,\n    forecast_croston, forecast_sba, forecast_tsb\n)\n# All seven methods now use the documented shared initialization conventions.""")
        # Existing predict router calls Croston's old `sba` flag; replace it.
        source="""FORECAST_FUNCTIONS = {'Average': forecast_average, 'Naive': forecast_naive,\n                      'SeasonalNaive': forecast_seasonal_naive, 'SES': forecast_ses,\n                      'Croston': forecast_croston, 'SBA': forecast_sba, 'TSB': forecast_tsb}\n\ndef predict(model, train, horizon, alpha=None):\n    kwargs = {} if alpha is None else {'alpha': alpha}\n    if model not in FORECAST_FUNCTIONS:\n        raise ValueError('Unknown model: ' + model)\n    return FORECAST_FUNCTIONS[model](train, horizon, **kwargs)\n\nalpha_values = list(EXPERIMENT_CONFIG.alphas)\nall_models = list(FORECAST_FUNCTIONS)\nworkflows = {category: all_models.copy() for category in ['Smooth','Erratic','Intermittent','Lumpy']}\ndef configurations(category):\n    for model in workflows[category]:\n        for alpha in (alpha_values if model in ['SES','Croston','SBA','TSB'] else [None]):\n            yield model, alpha\nprint('Candidate settings per category:', len(list(configurations('Smooth'))))"""
        set_cell(nb,17,source)
    elif name=="02B_full_statistical_benchmark.ipynb":
        set_cell(nb,5,"""from src.forecasting.statistical_models import forecast_matrix\ndef forecast_all(history, model, alpha=.2, horizon=H):\n    kwargs = {'alpha': alpha} if model in ('SES','Croston','SBA','TSB') else {}\n    return forecast_matrix(model, history, horizon, **kwargs).astype(np.float32)""")
    elif name in {"03_global_machine_learning_forecasting.ipynb","03B_hurdle_lightgbm.ipynb"}:
        target=9 if name.startswith("03_global") else 6
        set_cell(nb,target,"""# Reload so an already-running Jupyter kernel cannot retain an older feature schema.
from importlib import reload
import src.features.feature_engineering as feature_engineering
feature_engineering = reload(feature_engineering)

def snapshot_features(history):
    output = feature_engineering.historical_snapshot(history)
    required = {'last_28_vs_prev_28'}
    missing = required.difference(output)
    if missing:
        raise RuntimeError(f'Shared feature schema is missing: {sorted(missing)}')
    return output

# Series code is a categorical indicator. Demand type is not used as a feature.
series_index = np.arange(S, dtype=np.int32)

def make_origin_rows(origin, with_target=True):
    # Create S x horizon rows using history strictly before origin.
    assert MIN_HISTORY <= origin <= T-HORIZON
    snap = snapshot_features(Y[:, :origin])
    rows = {key: np.repeat(value, HORIZON) for key, value in snap.items()}
    rows['series_code'] = np.repeat(series_index, HORIZON)
    rows['horizon'] = np.tile(np.arange(1, HORIZON+1, dtype=np.int16), S)
    future_dates = dates[origin:origin+HORIZON]
    rows['day_of_week'] = np.tile(future_dates.dayofweek.to_numpy(dtype=np.int8), S)
    rows['month'] = np.tile(future_dates.month.to_numpy(dtype=np.int8), S)
    frame = pd.DataFrame(rows)
    if with_target:
        return frame, Y[:,origin:origin+HORIZON].reshape(-1)
    return frame

example_x, example_y = make_origin_rows(test_origin)
display(example_x.head(7))
print('28 future prediction rows per series:', len(example_x), 'total')""")
        stat=19 if name.startswith("03_global") else 15
        set_cell(nb,stat,"""from src.forecasting.statistical_models import forecast_ses, forecast_sba, forecast_tsb\nALPHAS = EXPERIMENT_CONFIG.alphas\n\ndef statistical_rate(history, method, alpha):\n    fn = {'SES': forecast_ses, 'SBA': forecast_sba, 'TSB': forecast_tsb}[method]\n    return float(fn(history, 1, alpha=alpha)[0])\n\nbaseline_pred={}\nstat_param_rows=[]\nfor method in ('SES','SBA','TSB'):\n    result=np.zeros((S,HORIZON),dtype=np.float32)\n    for i in range(S):\n        errors=[]\n        for alpha in ALPHAS:\n            fold_errors=[]\n            for origin in validation_origins:\n                rate=statistical_rate(Y[i,:origin],method,alpha)\n                fold_errors.append(np.abs(Y[i,origin:origin+HORIZON]-rate).mean())\n            errors.append(float(np.mean(fold_errors)))\n        best_alpha=ALPHAS[int(np.argmin(errors))]\n        rate=statistical_rate(Y[i,:test_origin],method,best_alpha)\n        result[i,:]=rate\n        stat_param_rows.append({'series_id':ids[i],'model':method,'alpha':best_alpha,'cv_MAE':min(errors)})\n    baseline_pred[method]=result\n    print(method,'test forecast complete')\nstat_params=pd.DataFrame(stat_param_rows)\nprint('Validation-selected alpha counts:')\ndisplay(stat_params.groupby(['model','alpha']).size().rename('series_count').reset_index())""")
    elif name=="01_data_exploration_full_export.ipynb":
        # Keep charts/exports, but route classification helpers through shared code.
        set_cell(nb,11,"""from src.data.demand_classification import classify_demand\n\n# Preserve every original identifier/metadata column alongside the derived statistics.\nid_columns = [column for column in sales.columns if column not in day_cols]\nsummary = pd.concat([sales[id_columns].copy(), stats], axis=1)\n\ndef demand_type(row):\n    return classify_demand(row[train_cols].to_numpy(dtype=float),\n                           leading_zero_policy='active')\n\nsummary['demand_type'] = sales.apply(demand_type, axis=1)\nassert len(summary) == len(sales)\nprint(summary['demand_type'].value_counts())""")
    elif name=="03C_large_scale_lightgbm_hurdle.ipynb":
        notice="\n**Corrected protocol:** training-size and hyperparameter choices require historical validation; the previously explored final 28 days are descriptive only. Seen-series and unseen-series results must be exported separately.\n"
        if notice not in nb["cells"][0]["source"]: nb["cells"][0]["source"].append(notice)

    # Complete shared-module integration: notebooks retain orchestration only.
    if name=="02_forecasting_4_workflows_10_seeds.ipynb":
        set_cell(nb,19,"""from src.evaluation.metrics import mae, mase, mean_signed_bias, wape

def bias(actual, predicted):
    return mean_signed_bias(actual, predicted)

def aggregate_wape(table):
    denominator = table['actual_units'].sum()
    return table['abs_error_units'].sum() / denominator if denominator > 0 else np.nan

def evaluate_setting(history, model, alpha):
    cv = TimeSeriesSplit(n_splits=n_folds, test_size=validation_days)
    errors = []
    for train_idx, val_idx in cv.split(history):
        train, actual = history[train_idx], history[val_idx]
        errors.append(mae(actual, predict(model, train, len(actual), alpha)))
    return float(np.mean(errors))""")
    if name=="03_global_machine_learning_forecasting.ipynb":
        set_cell(nb,13,"""from src.forecasting.lightgbm_direct import fit_predict_frames
BASE_FEATURES = ['series_code','horizon','lag_1','lag_7','lag_14','lag_28','lag_56',
                 'mean_7','std_7','zero_ratio_7','mean_28','std_28','zero_ratio_28',
                 'mean_56','std_56','zero_ratio_56','last_28_vs_prev_28']
CAL_FEATURES = ['day_of_week','month']

def data_before(origin, calendar_on):
    pairs=[make_origin_rows(o) for o in training_origins_before(origin)]
    feature_names=BASE_FEATURES+(CAL_FEATURES if calendar_on else [])
    X=pd.concat([x for x,_ in pairs],ignore_index=True)[feature_names]
    X['series_code']=X['series_code'].astype('category')
    return X,np.concatenate([y for _,y in pairs])

def fit_and_forecast(origin, calendar_on):
    X_train,y_train=data_before(origin,calendar_on)
    feature_names=list(X_train.columns)
    X_future=make_origin_rows(origin,with_target=False)[feature_names].copy()
    X_future['series_code']=pd.Categorical(X_future['series_code'],categories=X_train['series_code'].cat.categories)
    prediction,model=fit_predict_frames(X_train,y_train,X_future,
        params={'random_state':SEED},categorical_features=['series_code'])
    return prediction.reshape(S,HORIZON),model,len(X_train)

print('Features WITHOUT calendar:',BASE_FEATURES)
print('Extra calendar features:',CAL_FEATURES)""")
    if name=="03B_hurdle_lightgbm.ipynb":
        set_cell(nb,9,"""from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames
BASE_FEATURES = ['series_code','horizon','lag_1','lag_7','lag_14','lag_28','lag_56',
                 'mean_7','std_7','zero_ratio_7','mean_28','std_28','zero_ratio_28',
                 'mean_56','std_56','zero_ratio_56','last_28_vs_prev_28']
CAL_FEATURES=['day_of_week','month'];FEATURES=BASE_FEATURES+CAL_FEATURES

def data_before(origin):
    pairs=[make_origin_rows(o) for o in training_origins_before(origin)]
    X=pd.concat([x[FEATURES] for x,_ in pairs],ignore_index=True)
    X['series_code']=X['series_code'].astype('category')
    return X,np.concatenate([y for _,y in pairs])

def features_at(origin,train_X):
    X=make_origin_rows(origin,with_target=False)[FEATURES].copy()
    X['series_code']=pd.Categorical(X['series_code'],categories=train_X['series_code'].cat.categories)
    return X

MODEL_ARGS=dict(n_estimators=250,learning_rate=.05,num_leaves=31,min_child_samples=70,
                colsample_bytree=.9,subsample=.9,subsample_freq=1,reg_lambda=1.,random_state=SEED,n_jobs=-1,verbosity=-1)
def fit_hurdle_and_direct(origin):
    X,y=data_before(origin);X_new=features_at(origin,X)
    result=fit_predict_hurdle_and_direct_frames(X,y,X_new,params=MODEL_ARGS,categorical_features=['series_code'])
    for key in ['hurdle','direct','probability','conditional']:
        result[key]=result[key].reshape(S,HORIZON)
    return result
print('Total features:',len(FEATURES),'| Train origin step:',ORIGIN_STEP)""")
    if name in {"03_global_machine_learning_forecasting.ipynb","03B_hurdle_lightgbm.ipynb"}:
        metrics_cell=21 if name.startswith("03_global") else 17
        set_cell(nb,metrics_cell,"""from src.evaluation.metrics import series_metric_records, aggregate_metric_records
truth=Y[:,test_origin:]
train=Y[:,:test_origin]
all_predictions={**baseline_pred,**ml_test_pred}
metric_rows=series_metric_records(truth,all_predictions,train,ids,label_types,
                                  underforecast_penalty=5.,overforecast_penalty=1.)
leaderboard=(metric_rows.groupby('model').apply(aggregate_metric_records,include_groups=False)
             .reset_index().sort_values('MAE'))
by_type=(metric_rows.groupby(['demand_type','model']).apply(aggregate_metric_records,include_groups=False)
         .reset_index())
display(leaderboard.round(4))
display(by_type[['demand_type','model','series','MAE','MASE','RMSSE','WAPE_pct','bias']].round(4))
print('Validation-selected ML:',selected_ml,'| Test MAE:',round(float(leaderboard.set_index('model').loc[selected_ml,'MAE']),4))""")
    if name=="03C_large_scale_lightgbm_hurdle.ipynb":
        set_cell(nb,5,"""from importlib import reload
import src.features.feature_engineering as feature_engineering
feature_engineering=reload(feature_engineering)

calendar_path=PROJECT/'data'/'raw'/'calendar.csv'
if calendar_path.exists():
    cal=pd.read_csv(calendar_path,usecols=['d','date']).drop_duplicates('d').set_index('d')
    dates=pd.DatetimeIndex(pd.to_datetime(cal.reindex(days)['date']))
    assert not dates.isna().any()
else:
    dates=pd.date_range('2011-01-29',periods=T,freq='D')

def features_for(ids,origin):
    hist=Y[ids,:origin]
    assert origin>=MIN_HISTORY and origin+H<=T
    snap=feature_engineering.historical_snapshot(hist)
    n=len(ids)
    data={'horizon':np.tile(np.arange(1,H+1,dtype=np.int16),n),
          'store':np.repeat(full.store_id.iloc[ids].to_numpy(),H),
          'department':np.repeat(full.item_id.iloc[ids].astype(str).str.split('_').str[:2].str.join('_').to_numpy(),H)}
    for key,value in snap.items():
        if key=='last_28_vs_prev_28':
            continue
        output_key=key.replace('zero_ratio_','zero_')
        data[output_key]=np.repeat(value,H)
    future=dates[origin:origin+H]
    data['weekday']=np.tile(future.dayofweek.to_numpy(dtype=np.int8),n)
    data['month']=np.tile(future.month.to_numpy(dtype=np.int8),n)
    return pd.DataFrame(data)

def prep(frame):
    for column in ['store','department']:
        frame[column]=frame[column].astype('category')
    return frame

FEATURES=['horizon','store','department','lag_1','lag_7','lag_14','lag_28','lag_56',
          'mean_7','zero_7','std_7','mean_28','zero_28','std_28','mean_56','zero_56','std_56','weekday','month']
assert TEST>MIN_HISTORY+H*4
all_origins=list(range(MIN_HISTORY,VALIDATION-H+1,STEP))
train_origins=all_origins[-MAX_ORIGINS:] if MAX_ORIGINS is not None else all_origins
assert max(train_origins)+H<=VALIDATION
print('Training origins:',train_origins)
check=features_for(eval_pos[:2],TEST);backup=Y[eval_pos[0],TEST]
Y[eval_pos[0],TEST]=backup+999
pd.testing.assert_frame_equal(check,features_for(eval_pos[:2],TEST))
Y[eval_pos[0],TEST]=backup
print('Future-sales leakage smoke test PASSED')""")
        set_cell(nb,7,"""from src.forecasting.lightgbm_hurdle import fit_predict_hurdle_and_direct_frames
from src.evaluation.metrics import mae, wape, mase, rmsse, mean_signed_bias, inventory_cost_proxy

def make_training(ids,forecast_origin):
    eligible=list(range(MIN_HISTORY,forecast_origin-H+1,STEP))
    origins=eligible[-MAX_ORIGINS:] if MAX_ORIGINS is not None else eligible
    assert max(origins)+H<=forecast_origin
    parts=[features_for(ids,o) for o in origins]
    targets=[Y[ids,o:o+H].ravel() for o in origins]
    return prep(pd.concat(parts,ignore_index=True))[FEATURES],np.concatenate(targets)

def evaluation_frame(train_X,origin,ids=eval_pos):
    frame=prep(features_for(ids,origin))[FEATURES]
    for column in ['store','department']:
        frame[column]=pd.Categorical(frame[column],categories=train_X[column].cat.categories)
    return frame

def train_predict(ntrain,forecast_origin):
    idx=permutation[:ntrain];start=time.time()
    X,targets=make_training(idx,forecast_origin);X_future=evaluation_frame(X,forecast_origin)
    params=dict(n_estimators=200,learning_rate=.05,num_leaves=31,min_child_samples=80,
                reg_lambda=1.,random_state=SEED,n_jobs=-1,verbosity=-1)
    result=fit_predict_hurdle_and_direct_frames(X,targets,X_future,params=params,
                                                categorical_features=['store','department'])
    direct=result['direct'].reshape(len(eval_pos),H);hurdle=result['hurdle'].reshape(len(eval_pos),H)
    elapsed=time.time()-start
    del X,X_future,targets,result;gc.collect()
    return direct,hurdle,elapsed

truth=Y[eval_pos,VALIDATION:VALIDATION+H]
results=[];forecasts={}
for ntrain in TRAINING_SIZES:
    print('Training on',ntrain,'series...')
    direct,hurdle,elapsed=train_predict(ntrain,VALIDATION)
    for model,prediction in [('Direct',direct),('Hurdle',hurdle)]:
        results.append({'training_series':ntrain,'model':model,'eval_series':len(eval_pos),
                        'MAE':mae(truth,prediction),'WAPE_pct':100*wape(truth,prediction),
                        'bias':mean_signed_bias(truth,prediction),
                        'cost_proxy':inventory_cost_proxy(truth,prediction,UNDER,OVER),
                        'seconds_for_both_models':elapsed})
        forecasts[(ntrain,model)]=prediction
summary=pd.DataFrame(results);display(summary.round(4))

selected_row=summary.sort_values(['MAE','training_series','model']).iloc[0]
selected_n=int(selected_row.training_series);selected_model=str(selected_row.model)
print('Validation-selected final configuration:',selected_n,selected_model)
final_direct,final_hurdle,final_elapsed=train_predict(selected_n,TEST)
final_truth=Y[eval_pos,TEST:TEST+H]
final_pred=final_direct if selected_model=='Direct' else final_hurdle
per_series=pd.DataFrame({'MASE':[mase(a,p,t) for a,p,t in zip(final_truth,final_pred,Y[eval_pos,:TEST])],
                         'RMSSE':[rmsse(a,p,t) for a,p,t in zip(final_truth,final_pred,Y[eval_pos,:TEST])]})
final_summary=pd.DataFrame([{'scenario':'unseen_product_store_combination','training_series':selected_n,
    'model':selected_model,'eval_series':len(eval_pos),'MAE':mae(final_truth,final_pred),
    'WAPE_pct':100*wape(final_truth,final_pred),'MASE':per_series.MASE.mean(),'RMSSE':per_series.RMSSE.mean(),
    'bias':mean_signed_bias(final_truth,final_pred),'cost_proxy':inventory_cost_proxy(final_truth,final_pred,UNDER,OVER),
    'seconds_for_both_models':final_elapsed}])
display(final_summary.round(4));final_summary.to_csv(OUT/'phase3c_validation_selected_final_test.csv',index=False)""")
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print("migrated",name)
