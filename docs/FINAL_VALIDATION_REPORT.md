# Final scientific validation implementation report

Date: 2026-11-01

> Supersession note: explored-period rankings below remain descriptive, but any earlier multi-origin result that reused a later hyperparameter choice is superseded by the nested rolling-origin protocol. Corrected multi-origin ML/DL values must be recalculated before scientific comparison.

## Implemented

- Fixed 400-series panel: 100 Smooth, 100 Erratic, 100 Intermittent, and 100 Lumpy series; no duplicate product-store pairs.
- Standard prediction schema and strict scenario/series/origin/day/horizon/actual alignment checks.
- Scenario A and Scenario B definitions plus unseen-series isolation assertion.
- Corrected Notebook 3D with explicit fast/research modes, 750-step research configuration, smaller N-HiTS candidate, GPU diagnostics, validation-only configuration search, both DeepAR distributions, predictive-mean labeling, checkpoints, metadata, and optional six-origin execution.
- Notebook 4 with validation-tuned statistical forecasts, chronological Direct/Hurdle refits, strict 11-model coverage, diagnostic breakdowns, cluster bootstrap comparisons, manifests, and explicit incomplete status.
- Six non-overlapping statistical backtest origins with tuning windows strictly before each evaluation origin.

## Actually executed

The explored-final matched panel contained all eleven models. Mean daily MAE:

| Model | MAE |
|---|---:|
| Direct LightGBM | 2.0392 |
| Hurdle LightGBM | 2.0454 |
| TSB | 2.0682 |
| SES | 2.0698 |
| SBA | 2.2018 |
| Croston | 2.2671 |
| Average | 2.4428 |
| Seasonal Naive | 2.5064 |
| Naive | 2.6848 |
| DeepAR (20-step smoke) | 3.0004 |
| N-HiTS (20-step smoke) | 4.4305 |

These are descriptive results on a repeatedly inspected period. The neural rows are undertrained smoke results and are not evidence about research-trained models. Scenario B was not ranked because no standardized unseen-series predictions exist.

Cluster bootstrap intervals crossed zero for TSB versus Direct LightGBM and Direct versus Hurdle, so numerical ordering alone is not evidence of significance.

## Remaining work

- Run Notebook 3D in research mode, preferably with CUDA-enabled PyTorch after user-controlled installation, then rerun Notebook 4.
- Execute matched ML and DL forecasts at every predeclared historical origin. The ML pipeline has passed one real-data origin (22,400 matched rows); this is not the six-origin result.
- Produce Scenario-B standardized predictions for Direct/Hurdle/DL models.
- Run full-mode rather than fast-mode DL tuning before scientific model claims.
