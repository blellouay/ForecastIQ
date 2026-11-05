# Test execution report

- Command: `.venv\Scripts\python.exe -m pytest -q`
- Result: **146 passed** (latest final-validation run: 0.83 seconds with workspace-local pytest temp directory) on 2026-11-01.
- Coverage exercised: statistical edge cases and scalar/matrix equivalence; preparation; classification; feature calculations; future-mutation leakage; metrics; backtesting; matched observations; deterministic model selection.
- Real-data smoke test: loaded 10 M5 series with 1,913 days, forecast the 28-day horizon using shared SBA, classified demand, and wrote a hashed manifest; passed.
- Notebook validation: all seven notebook JSON files loaded and every Python code cell compiled; passed.
- Unified Notebook 4 execution: passed for seven statistical models plus matched Direct and Hurdle LightGBM predictions; NHITS/DeepAR explicitly reported as not rerun.
- Not executed: full 30,490-series recalculation, full LightGBM scaling study, or prospective multi-origin M5 benchmark. Existing metrics are not relabeled as corrected results.
