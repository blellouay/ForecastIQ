# Methodology

Demand type is descriptive, not a model label. ForecastIQ computes average demand interval (ADI) and squared coefficient of variation (CV²) from observed positive sales and classifies series as Smooth, Erratic, Intermittent, or Lumpy. The first positive sale is not treated as a verified launch date.

Statistical forecasts use only history available at each origin. Saved LightGBM inference requires an exact snapshot, ordered feature schema, and—where applicable—the frozen training `series_code`. N-HiTS full-panel inference uses `nf.predict(df=...)` with dates and sales truncated through the requested origin. No model is retrained during agent inference.

Prices exist in M5 but are not currently supplied to integrated forecasting models because operational future-price availability has not been verified.
