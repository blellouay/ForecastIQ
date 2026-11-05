# Changelog

## 2026-11-01 — nested look-ahead correction

- Replaced isolated-fold DL selection with complete-fold mean-MAE selection and stable configuration IDs.
- Added historically valid inner folds per outer origin, resumable hash-keyed validation caching, missing-fold rejection, and nested selection history.
- Expanded TSB tuning to independent alpha/beta grids and preserved statistical selections per outer origin.
- Added two-stage forecast-origin/product-cluster bootstrap uncertainty with matched population counts.
- Added regression tests for chronology, future-mutation invariance, aggregation, missing folds, alignment, and reproducibility; 160 tests pass.

## 2026-11-01 — DL completion pipeline

- Added explicit 20-step fast and 750-step research neural modes, memory-conscious N-HiTS, both DeepAR distributions, CUDA diagnostics, checkpoints, and training metadata.
- Added chronological Direct/Hurdle multi-origin training, strict completeness/coverage validation, and unified diagnostic breakdowns.
- Updated Notebooks 3D and 4 to keep partial experiments out of complete leaderboards.
- Verified 151 tests and a 22,400-row real-data matched ML smoke backtest.

## 2026-11-01

- Added canonical statistical forecasts and explicit leading-zero conventions.
- Added shared preparation, demand classification, leakage-safe features, metrics, comparison, backtesting, uncertainty, model selection, configuration, manifests, and LightGBM modules.
- Added nine requested test files and notebook migration tooling.
- Backed up and migrated the six requested notebooks to shared imports.
- Added audit, scientific-validation, test, and reproducibility documentation.
- Migrated Notebook 2 metrics and Notebooks 3A/3B/3C feature generation, LightGBM fitting, and evaluation to shared `src` APIs.
- Added deterministic evaluation panel, standardized record contract, corrected DL notebook, unified Notebook 4, six-origin statistical backtesting, cluster paired uncertainty, and final validation report.
