"""ForecastIQ Streamlit interface for the LangGraph forecasting workflow.

Run from repository root: python -m streamlit run dashboard/app.py
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.logic import (
    read_catalog, parse_m5_id, make_request, read_history,
    forecast_table, selection_frame, extract_ai_part, filter_catalog,
    model_availability_frame,
)

st.set_page_config(page_title='ForecastIQ | Demand Forecasting', page_icon='📈', layout='wide')
st.title('ForecastIQ')
st.caption('Agentic AI for Adaptive Retail Demand Forecasting · 28-day horizon · Python-owned predictions')

@st.cache_data(show_spinner=False)
def catalog_cached(path: str, modified: float):
    return read_catalog(path)

@st.cache_data(show_spinner=False)
def history_cached(path: str, modified: float, series_id: str, origin: int):
    return read_history(path, series_id, origin, days=90)

with st.sidebar:
    st.header('Forecast settings')
    csv_path = st.text_input('M5 sales CSV', value='data/raw/sales_train_validation.csv',
                             help='Existing sales_train_validation.csv or another M5-format file')
    csv_file = Path(csv_path).expanduser()
    catalog = {}
    try:
        if csv_file.is_file():
            catalog = catalog_cached(str(csv_file.resolve()), csv_file.stat().st_mtime)
        else:
            st.warning('CSV not found. Correct the path to load available products.')
    except Exception as exc:
        st.error(f'Unable to read M5 IDs: {exc}')

    new_product = st.checkbox('Forecast a new product with no sales history', value=False)
    product_filter = st.selectbox('Product compatibility',
                                  ['All products','LightGBM-compatible products',
                                   'N-HiTS-compatible products'])
    if catalog:
        try:
            catalog = filter_catalog(catalog, product_filter)
        except Exception as exc:
            st.warning(f'Compatibility filter unavailable: {exc}')
    if new_product:
        item = st.text_input('New M5-style product ID', value='FOODS_1_999',
                             help='Metadata identifier only; no sales history is fabricated.')
        store = st.selectbox('Store', ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3'])
        category = item.split('_')[0] if item else ''
        department = '_'.join(item.split('_')[:2]) if item else ''
        product_metadata={'category':category,'department':department,'store':store}
    else:
        item = st.selectbox('Product', list(catalog), index=0 if catalog else None,
                            placeholder='Choose an available product') if catalog else None
        store = st.selectbox('Store', catalog.get(item, []), index=0 if item else None) if item else None
        product_metadata=None
    model = st.selectbox('Model selection',
                         ['auto','TSB','SBA','SES','Croston','Average','Naive','SeasonalNaive',
                          'DirectLightGBM','HurdleLightGBM','GeneralizableDirectLightGBM',
                          'GeneralizableHurdleLightGBM','NHITS_small','NHITS_scalable','DeepAR'])
    if model == 'NHITS_small':
        st.info('Saved N-HiTS supports supplied-history CPU inference from origin d_1857 onward; it is not auto-ranked.')
        origin = st.number_input('Historical forecast origin (d_N)', min_value=1, max_value=1913,
                                 value=1857, step=28)
    elif model == 'NHITS_scalable':
        st.info('Full-panel N-HiTS uses the exact snapshot for d_1801, d_1829, d_1857, or d_1885 and is eligible for matched auto-ranking.')
        origin = st.number_input('Historical forecast origin (d_N)', min_value=1, max_value=1913,
                                 value=1885, step=28)
    else:
        origin = st.number_input('Historical forecast origin (d_N)', min_value=1, max_value=1913,
                                 value=1885, step=28)
    use_groq = st.checkbox('AI explanation via Groq', value=False,
                           help='Opt-in external call. Requires GROQ_API_KEY and may be subject to provider limits.')
    if use_groq and not os.getenv('GROQ_API_KEY'):
        st.warning('GROQ_API_KEY not found; offline explanation will be used.')
    nhits_dir = st.text_input('Saved N-HiTS directory', value='models/neuralforecast/nhits_small')
    nhits_scalable_root = st.text_input('Full M5 N-HiTS snapshot root', value='models/nhits_full_m5/full_all')
    registry = st.text_input('LightGBM registry (optional)', value='models/registry.json')
    run = st.button('Generate forecast', type='primary', use_container_width=True,
                    disabled=(not item or not store or (not new_product and not catalog)))

if 'forecastiq_result' not in st.session_state:
    st.session_state['forecastiq_result'] = None

if run:
    st.session_state['forecastiq_result'] = None
    try:
        request = make_request(item, store, model)
        from src.agents.conversational import build_conversational_workflow
        from src.features.m5_lightgbm import feature_provider
        provider = 'groq' if use_groq and os.getenv('GROQ_API_KEY') else 'offline'
        # The saved graph remains the source of truth for model selection and predictions.
        graph = build_conversational_workflow(llm_provider=provider,
                     ml_feature_provider=feature_provider,
                     llm_model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b') if provider == 'groq' else None)
        graph_input = {
            'request': request,
            'forecast_origin_day': int(origin),
        }
        if new_product:
            graph_input.update(sales_history=[],product_metadata=product_metadata,
                               comparable_sales_path=str(csv_file))
        else:
            graph_input['m5_sales_path']=str(csv_file)
        if Path(nhits_dir).is_dir():
            graph_input['nhits_checkpoint_path'] = nhits_dir
        if Path(nhits_scalable_root).is_dir():
            graph_input['nhits_scalable_root'] = nhits_scalable_root
        if Path(registry).is_file():
            graph_input['ml_registry_path'] = registry
        with st.spinner('Running forecasting workflow…'):
            conversation = graph.invoke(graph_input)
        st.session_state['forecastiq_result'] = {
            'conversation': conversation, 'origin': int(origin),
            'series_id': f'{item}_{store}_validation', 'csv_path': str(csv_file),
        }
    except Exception as exc:
        st.error(f'Forecast request failed: {type(exc).__name__}: {exc}')

saved = st.session_state['forecastiq_result']
if saved is None:
    st.info('Choose a product and store in the sidebar, then select **Generate forecast**.')
    st.stop()

conversation = saved['conversation']
result = conversation.get('forecast_result_state') or {}
if conversation.get('status') != 'completed' or result.get('status') != 'completed':
    st.error(conversation.get('error') or result.get('error') or conversation.get('response') or 'Forecast failed')
    st.stop()

try:
    forecast = forecast_table(result, saved['origin'])
except Exception as exc:
    st.error(f'Invalid forecast output: {exc}')
    st.stop()

payload = result.get('forecast_result') or {}
selected = result.get('selected_model') or result.get('model') or 'Unknown'
summary_total = float(forecast['forecast'].sum())

st.subheader(f'Forecast · {saved["series_id"]}')
a, b, c, d = st.columns(4)
a.metric('Selected model', selected)
b.metric('28-day expected demand', f'{summary_total:.2f} units')
c.metric('Mean per day', f'{summary_total/28:.3f} units')
d.metric('Demand type', (result.get('diagnostics') or {}).get('demand_type') or 'Not available')
st.info(f"Forecasting scenario: {(result.get('forecasting_scenario') or {}).get('scenario','unknown')} · "
        f"Historical observations: {(result.get('forecasting_scenario') or {}).get('historical_observations','unknown')}")

st.caption(f'Forecast origin d_{saved["origin"]} · Target d_{saved["origin"]+1} to d_{saved["origin"]+28} · Expected values, not guaranteed sales')

try:
    hist = history_cached(str(Path(saved['csv_path']).resolve()), Path(saved['csv_path']).stat().st_mtime,
                          saved['series_id'], saved['origin'])
    observed = hist.rename(columns={'actual':'Observed sales'}).set_index('day')
    predicted = forecast.rename(columns={'forecast':'Forecast'}).set_index('day')
    chart = observed.join(predicted, how='outer')
    st.subheader('Observed sales and forecast')
    st.line_chart(chart, x_label='M5 day number', y_label='Units per day')
except Exception as exc:
    st.warning(f'Historical sales chart unavailable: {exc}')
    st.subheader('Forecast')
    st.line_chart(forecast.set_index('day'), x_label='M5 day number', y_label='Expected units')

left, right = st.columns([1, 1])
with left:
    st.subheader('Historical model evidence')
    table = selection_frame(result)
    if not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info('Historical comparison is unavailable for this explicitly selected model.')
    st.write(result.get('selection_reason') or 'Model explicitly requested, or no selection details provided.')
    excluded = (result.get('selection_evidence') or {}).get('ml_excluded')
    if excluded:
        st.warning(f'ML models excluded: {excluded}')

st.subheader('Model Availability')
availability = model_availability_frame(result)
if availability.empty:
    st.info('Model availability was not reported for this request.')
else:
    st.dataframe(availability, use_container_width=True, hide_index=True)

with right:
    st.subheader('Forecast explanation')
    response = str(conversation.get('response') or '')
    st.write(response or 'No explanation returned.')
    explanation, llm_status = extract_ai_part(response, conversation)
    st.caption(f'AI explanation: {llm_status} · Groq is optional and does not calculate forecasts')
    if not explanation and use_groq:
        st.info('No accepted AI commentary was returned; the numerical forecast remains valid.')

st.subheader('Forecast by day')
st.dataframe(forecast.rename(columns={'day':'M5 day', 'forecast':'Expected units'}),
             hide_index=True, use_container_width=True)
st.download_button('Download forecast CSV',
                   forecast.to_csv(index=False).encode('utf-8'),
                   file_name=f'forecastiq_{selected}_{saved["origin"]}_forecast.csv',
                   mime='text/csv')
with st.expander('Technical metadata'):
    st.json({'model': selected, 'origin': saved['origin'],
             'selection_evidence': result.get('selection_evidence') or {},
             'checkpoint_metadata': payload.get('checkpoint_metadata') or {},
             'warnings': 'Historical predictions are not guaranteed sales. Past data do not reveal stockouts or true launch dates.'})
