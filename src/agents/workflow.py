"""ForecastIQ deterministic, inference-only LangGraph pipeline.

No LLM, model training, or future-demand access. Model choice is a transparent policy,
NOT a claim of CV-based optimality. Each call is an independent graph invocation.
"""
from __future__ import annotations
import csv
import math
import re
from pathlib import Path
from typing import Any, TypedDict

from src.agent_tools import ToolError, describe_history, forecast_statistical
from .model_router import select_by_rolling_validation, demand_profile
from .ml_routing import ML_NAMES, route_with_saved_ml, predict_snapshot
from .nhits_inference import (MODEL_NAME as NHITS_NAME, SCALABLE_MODEL_NAME,
                              NHITSInferenceError, predict_nhits, predict_nhits_scalable)
from .model_availability import assess_model_availability
from .generalizable_routing import NAMES as GENERAL_NAMES, predict_generalizable_snapshot
from src.forecasting.cold_start import detect_forecasting_scenario, comparable_product_forecast

ALLOWED_MODELS = {"Average", "Naive", "SeasonalNaive", "SES", "Croston", "SBA", "TSB"}
DEEPAR_NAME = 'DeepAR'
DAY_RE = re.compile(r"^d_(\d+)$")


class ForecastState(TypedDict, total=False):
    series_id: str
    sales_history: list[float]
    m5_sales_path: str
    forecast_origin_day: int
    model: str
    horizon: int
    alpha: float
    validation_folds: int
    min_train_days: int
    ml_registry_path: str
    generalizable_registry_path: str
    nhits_checkpoint_path: str
    nhits_scalable_root: str
    diagnostics: dict[str, Any]
    selected_model: str
    selection_reason: str
    selection_evidence: dict[str, Any]
    forecast_result: dict[str, Any]
    explanation: str
    status: str
    error: str
    model_availability: list[dict[str, Any]]
    forecasting_scenario: dict[str, Any]
    product_metadata: dict[str, Any]
    comparable_sales_path: str


def _read_m5_history(path: str, series_id: str, origin_day: int | None) -> list[float]:
    """Read one series from an original M5 wide-format sales CSV, no future columns."""
    file = Path(path)
    if not file.is_file():
        raise ToolError(f"M5 sales CSV not found: {file}")
    with file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ToolError("M5 wide CSV needs an 'id' column")
        days = sorted((int(match.group(1)), name)
                      for name in reader.fieldnames if (match := DAY_RE.fullmatch(name)))
        if not days:
            raise ToolError("M5 wide CSV has no daily d_N columns")
        if origin_day is None:
            raise ToolError("forecast_origin_day is required for M5 CSV to prevent accidental future-data use")
        if isinstance(origin_day, bool) or not isinstance(origin_day, int) or origin_day < 1:
            raise ToolError("forecast_origin_day must be a positive integer")
        selected = [(day, col) for day, col in days if day <= origin_day]
        if not selected or selected[-1][0] != origin_day or selected[0][0] != 1 or [d for d,_ in selected] != list(range(1, origin_day+1)):
            raise ToolError("Requested history dates are unavailable or non-contiguous")
        matching = [row for row in reader if row['id'] == series_id]
        if len(matching) != 1:
            raise ToolError(f"Expected exactly one M5 row for id={series_id!r}, found {len(matching)}")
        try:
            return [float(matching[0][col]) for _,col in selected]
        except (ValueError, TypeError) as exc:
            raise ToolError("M5 sales values must be numeric") from exc


def load_history(state: ForecastState) -> dict[str, Any]:
    try:
        series_id = state.get("series_id")
        if not isinstance(series_id, str) or not series_id.strip():
            raise ToolError("series_id is required")
        has_inline = "sales_history" in state
        has_csv = bool(state.get("m5_sales_path"))
        if has_inline == has_csv:
            raise ToolError("Supply exactly one source: sales_history or m5_sales_path")
        if has_csv:
            history = _read_m5_history(state['m5_sales_path'], series_id, state.get('forecast_origin_day'))
        else:
            history = state['sales_history']
            if state.get('forecast_origin_day') is not None and history:
                raise ToolError("For inline sales_history, omit forecast_origin_day; history is as-of supplied cutoff")
        # Reuse Step 1 validation before allowing any downstream calls.
        if history: describe_history(series_id, history)
        elif not state.get('product_metadata'): raise ToolError('No-history forecasts require product_metadata')
        return {'sales_history': list(history), 'status': 'history_loaded', 'error': ''}
    except (ToolError, OSError, csv.Error, TypeError, ValueError) as exc:
        return {'status': 'error', 'error': str(exc)}


def analyze_history(state: ForecastState) -> dict[str, Any]:
    try:
        scenario=detect_forecasting_scenario(state['series_id'],state['sales_history'])
        if state['sales_history']:
            diagnostics = describe_history(state['series_id'], state['sales_history'])
            diagnostics.update(demand_profile(state['sales_history']))
        else: diagnostics={'series_id':state['series_id'],'observations':0,'demand_type':'Unknown'}
        return {'diagnostics': diagnostics,'forecasting_scenario':scenario,'status': 'analyzed'}
    except (ToolError, KeyError) as exc:
        return {'status': 'error', 'error': str(exc)}


def discover_models(state: ForecastState, feature_provider=None) -> dict[str, Any]:
    try:
        return {'model_availability': assess_model_availability(
            state['series_id'], state['sales_history'],
            state.get('forecast_origin_day') or len(state['sales_history']),
            state.get('horizon', 28), state.get('ml_registry_path'), feature_provider,
            state.get('nhits_checkpoint_path', 'models/neuralforecast/nhits_small'),
            state.get('generalizable_registry_path'),
            state.get('nhits_scalable_root', 'models/nhits_full_m5/full_all'))}
    except Exception as exc:
        return {'model_availability': [],
                'status':'error', 'error':f'Model discovery failed: {type(exc).__name__}: {exc}'}


def select_model(state: ForecastState, feature_provider=None) -> dict[str, Any]:
    model = state.get('model', 'TSB')
    scenario=(state.get('forecasting_scenario') or {}).get('scenario')
    if model=='auto' and scenario=='no_history_cold_start':
        return {'selected_model':'ColdStartComparable','selection_reason':'No history: transparent comparable-product shrinkage baseline selected; no validation superiority claim.','selection_evidence':{},'status':'selected'}
    if model=='auto' and scenario=='short_history':
        return {'selected_model':'Average','selection_reason':'Short history: Average selected as a conservative fallback; matched validation is unavailable.','selection_evidence':{},'status':'selected'}
    if model == 'auto':
        try:
            decision = route_with_saved_ml(
                series_id=state['series_id'], history=state['sales_history'],
                origin=state.get('forecast_origin_day') or len(state['sales_history']),
                horizon=state.get('horizon', 28),
                registry_path=state.get('ml_registry_path'), feature_provider=feature_provider,
                alpha=state.get('alpha', 0.2),
                n_folds=state.get('validation_folds', 3),
                min_train_days=state.get('min_train_days', 90),
                generalizable_registry_path=state.get('generalizable_registry_path'),
                nhits_scalable_root=state.get('nhits_scalable_root', 'models/nhits_full_m5/full_all'),
            )
            availability = state.get('model_availability') or []
            decision['dl_excluded'] = {
                row['model']: row['reason'] for row in availability
                if row.get('family') == 'deep_learning' and not row.get('selection_eligible')
            }
            decision['inference_only_models'] = [
                row['model'] for row in availability
                if row.get('inference_eligible') and not row.get('selection_eligible')
            ]
            return {'selected_model': decision['model'],
                    'selection_reason': decision['reason'],
                    'selection_evidence': decision, 'status': 'selected'}
        except (ToolError, KeyError, TypeError, ValueError) as exc:
            return {'status': 'error', 'error': str(exc)}
    if model == NHITS_NAME:
        if not state.get('forecast_origin_day'):
            return {'status':'error', 'error':'NHITS_small requires forecast_origin_day to verify the saved checkpoint date.'}
        return {'selected_model':model, 'selection_reason':'Caller explicitly selected saved N-HiTS checkpoint; no auto-selection or accuracy claim.', 'status':'selected'}
    if model == SCALABLE_MODEL_NAME:
        if not state.get('forecast_origin_day'):
            return {'status':'error', 'error':'NHITS_scalable requires an exact supported forecast_origin_day.'}
        return {'selected_model':model, 'selection_reason':'Caller explicitly selected the exact-origin full-panel N-HiTS checkpoint.', 'status':'selected'}
    if model == DEEPAR_NAME:
        return {'status':'error', 'error':'DeepAR has training checkpoints but no verified deployable inference bundle and provenance registry.'}
    if model in ML_NAMES:
        if not state.get('ml_registry_path') or feature_provider is None:
            return {'status':'error', 'error':'Saved ML needs ml_registry_path and an as-of feature provider; no retraining.'}
        return {'selected_model':model, 'selection_reason':'Caller explicitly selected a trusted saved ML model; no comparative claim.', 'status':'selected'}
    if model in GENERAL_NAMES:
        if not state.get('generalizable_registry_path'):
            return {'status':'error','error':'Generalizable LightGBM needs generalizable_registry_path.'}
        return {'selected_model':model,'selection_reason':'Caller explicitly selected a saved generalizable LightGBM model.', 'status':'selected'}
    if model not in ALLOWED_MODELS:
        return {'status': 'error', 'error': f'Model {model!r} unavailable in Step 3. Supported: {sorted(ALLOWED_MODELS)} or auto'}
    if 'model' in state:
        reason = 'Caller explicitly selected model; no model-performance claim.'
    else:
        reason = 'TSB default demonstration policy; NOT selected by cross-validation or proven best.'
    return {'selected_model': model, 'selection_reason': reason, 'status': 'selected'}


def generate_forecast(state: ForecastState, feature_provider=None) -> dict[str, Any]:
    try:
        if state['selected_model']=='ColdStartComparable':
            result=comparable_product_forecast(state['series_id'],state['product_metadata'],
                state['comparable_sales_path'],state['forecast_origin_day'],state.get('horizon',28))
        elif state['selected_model'] == NHITS_NAME:
            result = predict_nhits(
                series_id=state['series_id'],
                origin_day=state.get('forecast_origin_day'),
                horizon=state.get('horizon', 28),
                checkpoint_path=state.get('nhits_checkpoint_path', 'models/neuralforecast/nhits_small'),
                historical_sales=state['sales_history'],
            )
        elif state['selected_model'] == SCALABLE_MODEL_NAME:
            result = predict_nhits_scalable(
                state['series_id'], state.get('forecast_origin_day'), state['sales_history'],
                state.get('horizon', 28),
                state.get('nhits_scalable_root', 'models/nhits_full_m5/full_all'))
        elif state['selected_model'] in ML_NAMES:
            result = predict_snapshot(
                series_id=state['series_id'], history=state['sales_history'],
                origin=state.get('forecast_origin_day') or len(state['sales_history']),
                horizon=state.get('horizon', 28), model=state['selected_model'],
                registry_path=state['ml_registry_path'], feature_provider=feature_provider)
        elif state['selected_model'] in GENERAL_NAMES:
            result=predict_generalizable_snapshot(
                state['series_id'],state['sales_history'],state.get('forecast_origin_day') or len(state['sales_history']),
                state.get('horizon',28),state['selected_model'],state['generalizable_registry_path'])
        else:
            result = forecast_statistical(
                series_id=state['series_id'], sales_history=state['sales_history'],
                model=state['selected_model'], horizon=state.get('horizon', 28),
                alpha=state.get('alpha', 0.2))
        return {'forecast_result': result, 'status': 'forecasted'}
    except (ToolError, NHITSInferenceError, KeyError, TypeError, ValueError) as exc:
        return {'status': 'error', 'error': str(exc)}


def review_forecast(state: ForecastState) -> dict[str, Any]:
    try:
        result = state['forecast_result']
        horizon = state.get('horizon', 28)
        values = result['forecast']
        if (len(values) != horizon or result['horizon'] != horizon or
            result['series_id'] != state['series_id'] or
            result['model'] != state['selected_model'] or
            any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values) or
            not math.isclose(sum(values), result['total_expected_demand'], abs_tol=1e-6, rel_tol=1e-8)):
            raise ToolError('Forecast failed consistency and nonnegative checks')
        explanation = (f"{state['selected_model']} forecasts {result['total_expected_demand']:.2f} total units "
                       f"over {horizon} days for {state['series_id']}. "
                       f"{state['selection_reason']} "
                       "Sales history does not reveal stockouts or true product launch dates.")
        return {'status': 'completed', 'explanation': explanation}
    except (ToolError, KeyError, TypeError, ValueError) as exc:
        return {'status': 'error', 'error': str(exc)}


def _proceed(state: ForecastState) -> str:
    return 'end' if state.get('status') == 'error' else 'continue'


def build_workflow(ml_feature_provider=None, ml_registry_path=None,
                   generalizable_registry_path='models/generalizable_lightgbm/registry.json'):
    """Construct the actual LangGraph; import lazily to allow node unit testing."""
    try:
        from langgraph.graph import START, END, StateGraph
    except ImportError as exc:
        raise RuntimeError('Install LangGraph: python -m pip install "langgraph>=1,<2"') from exc
    graph = StateGraph(ForecastState)
    def configure(state):
        updates={}
        if not state.get('ml_registry_path') and ml_registry_path is not None:
            updates['ml_registry_path']=str(ml_registry_path)
        if not state.get('generalizable_registry_path') and generalizable_registry_path is not None:
            updates['generalizable_registry_path']=str(generalizable_registry_path)
        return updates
    graph.add_node('configure', configure)
    graph.add_node('load_history', load_history)
    graph.add_node('analyze_history', analyze_history)
    graph.add_node('discover_models', lambda state: discover_models(state, ml_feature_provider))
    graph.add_node('select_model', lambda state: select_model(state, ml_feature_provider))
    graph.add_node('generate_forecast', lambda state: generate_forecast(state, ml_feature_provider))
    graph.add_node('review_forecast', review_forecast)
    graph.add_edge(START, 'configure')
    graph.add_edge('configure', 'load_history')
    for previous, following in [('load_history','analyze_history'),('analyze_history','discover_models'),
                                ('discover_models','select_model'),
                                ('select_model','generate_forecast'),('generate_forecast','review_forecast')]:
        graph.add_conditional_edges(previous, _proceed, {'continue': following, 'end': END})
    graph.add_edge('review_forecast', END)
    return graph.compile()
