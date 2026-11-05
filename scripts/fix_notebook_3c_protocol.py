"""Correct Notebook 3C's validation/test separation without removing its charts."""
import json
from pathlib import Path

path=Path(__file__).resolve().parents[1]/"notebooks"/"03C_large_scale_lightgbm_hurdle.ipynb"
nb=json.loads(path.read_text(encoding="utf-8"))

def replace(cell,old,new):
    source="".join(cell["source"])
    if old not in source: raise RuntimeError(f"expected Notebook 3C text not found: {old[:50]}")
    cell["source"]=(source.replace(old,new)).splitlines(keepends=True)
    cell["execution_count"]=None; cell["outputs"]=[]

# Shared bootstrap insertion shifts original code cells by one.
setup,features,experiment,diagnostics=nb["cells"][3],nb["cells"][5],nb["cells"][7],nb["cells"][9]
replace(setup,"TRAINING_SIZES=[400,2000,5000] # optional: add 10000 and 30090",
              "TRAINING_SIZES=[400,2000,5000,10000] # maximum eligible size is appended below")
replace(setup,"N,T=Y.shape;TEST=T-H", "N,T=Y.shape;TEST=T-H;VALIDATION=TEST-H")
replace(setup,"TRAINING_SIZES=[s for s in TRAINING_SIZES if s<=len(permutation)]",
              "TRAINING_SIZES=sorted(set([s for s in TRAINING_SIZES if s<=len(permutation)]+[len(permutation)]))")
replace(features,"all_origins=list(range(MIN_HISTORY,TEST-H+1,STEP))",
                 "all_origins=list(range(MIN_HISTORY,VALIDATION-H+1,STEP))")
replace(features,"assert max(train_origins)+H<=TEST", "assert max(train_origins)+H<=VALIDATION")
replace(experiment,"def make_training(ids):", "def make_training(ids, forecast_origin):")
replace(experiment,"    for o in train_origins:",
                   "    eligible_origins=list(range(MIN_HISTORY,forecast_origin-H+1,STEP))\n    origins=eligible_origins[-MAX_ORIGINS:] if MAX_ORIGINS is not None else eligible_origins\n    assert max(origins)+H<=forecast_origin\n    for o in origins:")
replace(experiment,"def evaluation_frame(train_X, ids=eval_pos):\n    X=prep(features_for(ids,TEST))[FEATURES]",
                   "def evaluation_frame(train_X, origin, ids=eval_pos):\n    X=prep(features_for(ids,origin))[FEATURES]")
replace(experiment,"def train_predict(ntrain):", "def train_predict(ntrain, forecast_origin):")
replace(experiment,"X,targets=make_training(idx)\n    Xe=evaluation_frame(X)",
                   "X,targets=make_training(idx,forecast_origin)\n    Xe=evaluation_frame(X,forecast_origin)")
replace(experiment,"truth=Y[eval_pos,TEST:TEST+H]", "truth=Y[eval_pos,VALIDATION:VALIDATION+H]")
replace(experiment,"direct,hurdle,elapsed=train_predict(ntrain)", "direct,hurdle,elapsed=train_predict(ntrain,VALIDATION)")
append="""

# Select training population and model on historical validation only.
selected_row=summary.sort_values(['MAE','training_series','model']).iloc[0]
selected_n=int(selected_row.training_series);selected_model=str(selected_row.model)
print('Validation-selected final configuration:',selected_n,selected_model)

# Evaluate the selected configuration once on the previously explored final 28 days.
final_direct,final_hurdle,final_elapsed=train_predict(selected_n,TEST)
final_truth=Y[eval_pos,TEST:TEST+H]
final_pred=final_direct if selected_model=='Direct' else final_hurdle
final_error=final_pred-final_truth
train_history=Y[eval_pos,:TEST]
mae_scale=np.abs(np.diff(train_history,axis=1)).mean(axis=1)
rmsse_scale=np.square(np.diff(train_history,axis=1)).mean(axis=1)
per_series_mae=np.abs(final_error).mean(axis=1)
per_series_rmse=np.sqrt(np.square(final_error).mean(axis=1))
final_summary=pd.DataFrame([{
    'scenario':'unseen_product_store_combination','training_series':selected_n,'model':selected_model,
    'eval_series':len(eval_pos),'MAE':np.abs(final_error).mean(),
    'WAPE_pct':100*np.abs(final_error).sum()/final_truth.sum() if final_truth.sum() else np.nan,
    'MASE':np.nanmean(np.divide(per_series_mae,mae_scale,where=mae_scale>0)),
    'RMSSE':np.nanmean(np.divide(per_series_rmse,np.sqrt(rmsse_scale),where=rmsse_scale>0)),
    'bias':final_error.mean(),'cost_proxy':(UNDER*np.maximum(-final_error,0)+OVER*np.maximum(final_error,0)).sum(),
    'seconds_for_both_models':final_elapsed
}])
display(final_summary.round(4))
final_summary.to_csv(OUT/'phase3c_validation_selected_final_test.csv',index=False)
"""
experiment["source"].extend(append.splitlines(keepends=True))
replace(diagnostics,"summary.to_csv(OUT/'phase3c_scaling_overall.csv',index=False)",
                    "summary.to_csv(OUT/'phase3c_scaling_validation.csv',index=False)")
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print("corrected",path)
