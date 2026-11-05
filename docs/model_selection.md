# Model selection

Inference eligibility means a model can safely generate a forecast at the requested origin. Selection eligibility additionally requires comparable predictions for every required historical fold. The two concepts are intentionally distinct.

At final origin d_1885, automatic routing validates candidates at d_1801, d_1829, and d_1857. Every fold forecasts the next 28 observations for the same series. Mean absolute error is computed from actual per-series predictions, never copied from aggregate notebook metrics. The lowest mean MAE wins with a fixed deterministic tie order.

Missing artifacts, unknown frozen mappings, incompatible schemas, future-trained weights, wrong dates, or invalid predictions exclude only the affected candidate and record a reason. They never trigger fabricated IDs or silent retraining.
