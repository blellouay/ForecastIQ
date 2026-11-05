# Deep-learning completion reproducibility report

Date: 2026-11-01

## Audited environment

- Python 3.11.15 from the project `.venv`.
- NeuralForecast 3.3.0.
- PyTorch 2.15.0+cpu; CUDA build `None`; `torch.cuda.is_available()` false.
- `nvidia-smi` returned `Failed to initialize NVML: Not Found`.
- RAM: 15.71 GB.

The immediate CUDA cause is the CPU-only PyTorch wheel. Driver/NVML health must also be resolved before a CUDA wheel can work. No package was uninstalled or replaced automatically. Use the official PyTorch “Start Locally” selector for Windows + Pip + a supported CUDA platform, then run its generated command from `.venv` and restart the Jupyter kernel.

## Evidence executed

- 160 automated tests passed, including statistical equivalence/edge cases, nested chronology, missing-fold rejection, aggregation across folds, future-mutation invariance, leakage, metrics, panel reproducibility, scenario isolation, neural configuration, hierarchical uncertainty, and multi-origin completeness.
- Both updated notebooks compile cell-by-cell.
- A real-data Direct/Hurdle LightGBM smoke backtest at origin 1857 produced 22,400 rows: 400 series × 28 targets × 2 models. Both model blocks passed unique-key and matched-coverage checks.
- A real-data nested statistical smoke test at outer origin 1717 produced 78,400 matched rows. Its inner folds were 1633, 1661, and 1689; every 28-day validation target ended by 1717. It produced 21 candidate-level validation rows and 2,800 series/model selection records.
- The preserved 20-step neural artifacts contain all 11 explored-period models. Validation favored Negative Binomial DeepAR at the single fast origin, but its final-period bias remained strongly negative; no research conclusion is drawn.

## Exact execution

1. Open `03D_nhits_deepar.ipynb`, set `MODE='research'`, and run all cells. Leave `RUN_MULTI_ORIGIN=False` for configuration selection and the explored-period diagnostic.
2. After confirming resource use/checkpoints, set `RUN_MULTI_ORIGIN=True` and rerun the optional final cell to produce `phase3d_multi_origin_predictions.csv`.
3. Open `04_unified_forecasting_benchmark.ipynb`, set `RUN_ML_MULTI_ORIGIN=True`, and run all cells. Statistical and neural parameters are selected independently for every outer origin; LightGBM uses the documented predeclared fixed configuration.
4. Confirm `unified_multi_origin_coverage.csv` says `complete` for every intended model/origin before interpreting the multi-origin paired table.
5. Run `./.venv/Scripts/python.exe -m pytest -q`.

## Not yet executed

- The 750-step N-HiTS/DeepAR validation search.
- Six-origin ML and DL training.
- Multi-origin hierarchical confidence intervals involving unfinished ML/DL records.
- Scenario-B unseen-series forecasts.
- A prospective untouched test period (none is available in the current M5 horizon).

These omissions are surfaced as incomplete coverage and are never silently included in a full leaderboard.
