# ForecastIQ audit report

Audit date: 2026-11-01. The project directory is not an independent Git repository, so the six original notebooks were preserved in `notebooks_backup_20261101/` before migration.

## Findings

- Notebook 2 initialized Croston's interval with `first + 1`, Notebook 2B used one, and Notebook 3B again used `first + 1`. TSB probability initialization also differed.
- Notebook 2 trimmed leading zeros before validation, whereas the full benchmark and ML baselines generally preserved them.
- Statistical functions, metrics, features, and cost calculations were duplicated.
- Prediction exports did not share a single matched-observation schema.
- Notebook 3C compared training sizes on an already explored test window and had no separate validation stage.
- The ML snapshot features were fixed-origin and passed an inline future-mutation smoke test, but that guarantee was not reusable or automatically tested.
- The 1,913-day boundary and `d_1886`–`d_1913` evaluation window match the available M5 accuracy data.

## Resolution

Shared modules now define model conventions, preprocessing policies, origin-safe classification/features, metrics, alignment validation, backtesting, uncertainty, selection, and ML interfaces. Historical outputs produced by inconsistent definitions are retained as artifacts but are not treated as corrected results.
