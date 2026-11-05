# ForecastIQ architecture

ForecastIQ separates numerical forecasting from optional language generation. `src/agents/workflow.py` validates the request, loads only as-of history, classifies demand, evaluates model eligibility, routes by matched historical MAE, generates a forecast, and performs consistency checks. `src/agents/conversational.py` may then request a short qualitative explanation; the LLM cannot modify model selection or numbers.

Core layers are `src/data` for preparation and classification, `src/features` for origin-safe feature construction, `src/forecasting` for model implementations, `src/evaluation` for backtesting, and `src/agents` for orchestration. Registries bind saved artifacts to feature schemas, identities, origins, and training cutoffs.

The Streamlit application invokes the same graph used by examples and tests. It does not contain a parallel forecasting implementation.
