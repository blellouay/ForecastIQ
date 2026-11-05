# Scientific validation protocol

## Leakage prevention

Every forecast is constructed at an explicit origin. Historical lags and rolling statistics use `sales[:, :origin]`; future calendar fields are allowed only because dates are known in advance. Tests mutate all post-origin sales and assert unchanged features and classifications.

## Statistical conventions

The canonical definitions are in `src/forecasting/statistical_models.py`. Croston/SBA initialize positive size at the first event and interval at one. TSB initializes event probability at one on the first event and updates it daily thereafter. The default leading-zero policy is `preserve`; `active` is an explicit controlled alternative. The first positive sale is not claimed to be a launch date.

## Evaluation

The historical common window trains/validates through `d_1885` and evaluates 28 days (`d_1886`–`d_1913`). Because this test has already been inspected, it is labeled an explored benchmark. Stronger conclusions require six or more predeclared non-overlapping 28-day origins. MASE/RMSSE scales come only from training history; undefined denominators remain `NaN`. Global WAPE is computed from pooled absolute error and pooled actual demand.

Models may be compared only after `assert_matched_predictions` verifies identical series, origins, days, horizons, and actuals. Paired uncertainty resamples series/product clusters, not individual days. Numerical superiority alone is not statistical significance.

## Nested rolling-origin selection

Every historical evaluation origin is an outer origin. Its inner validation origins are generated solely from the history available at that time, with `inner_origin + 28 <= outer_origin`. Candidate scores are averaged across the same complete set of inner folds. Candidates with failed, missing, or duplicate folds are ineligible; ties are resolved by a stable hash of the complete configuration.

Statistical smoothing parameters are reselected inside each outer origin. TSB tunes demand-size `alpha` and occurrence-probability `beta` separately. The multi-origin LightGBM protocol currently uses one predeclared fixed configuration for each Direct and Hurdle model; this tuning budget of one avoids future-informed selection. Each DL outer origin performs its own resumable candidate search. A selection made for origin 1885 is never reused at origin 1717.

The validation cache key includes the frozen panel hash, mode, seed, full configuration ID, inner origin, outer origin, and requested training steps. Cached scores cannot cross outer-origin boundaries.

Multi-origin confidence intervals use a two-stage bootstrap: forecast origins are resampled first to represent shared temporal shocks, then product clusters are resampled within origins to preserve dependence across stores. Daily errors are reduced to series-origin MAE before resampling. Six origins provide weak temporal resolution, so these intervals remain approximate.

## Selection and business interpretation

Global, per-type, and per-series selection use the same candidates and validation data, deterministic ties, and an optional predeclared minimum improvement. The asymmetric cost is a forecast-error proxy, not an inventory simulation or monetary savings estimate.

## Seen versus unseen series

Seen-series evaluation includes a product-store series in training. Unseen-series evaluation excludes that combination while permitting its pre-origin context. Entirely unseen products are a stricter third case. Reports must not pool these populations.

## Deep-learning and compute protocol

`fast` is a 20-step smoke configuration at one validation origin. `research` is a predeclared 750-step configuration with a 168-day input, three validation origins, early stopping, checkpointing, and batches/windows sized conservatively for 4-GB VRAM. N-HiTS compares a smaller 128-unit configuration with the library default; DeepAR compares Student-t/robust scaling with Negative Binomial/identity scaling. Selection is by mean validation MAE, never the explored final period.

The audited interpreter is the project `.venv`, using `torch 2.15.0+cpu`; `torch.version.cuda` is `None`, and `nvidia-smi` could not initialize NVML. This identifies a CPU-only wheel and an unresolved driver/NVML issue. CUDA installation is intentionally user-controlled. The active Jupyter kernel must point to this same `.venv` before any package diagnosis is accepted.

## Remaining limitations

Recorded sales can be censored by stockouts. First-sale trimming is only a proxy. The current reference matrix implementation prioritizes mathematical equivalence over maximum full-panel speed. No untouched prospective final period remains in this dataset workflow.

## Unified panel and final validation status

`evaluation_panel.parquet` freezes the existing balanced pilot identifiers (100 per demand type), classification cutoff `d_1885`, final origin, and predeclared historical origins. Scenario A permits global training on evaluation-series history; Scenario B excludes evaluation product-store combinations from global-model training. Statistical models may use local history in both. Rankings from the two scenarios must remain separate.

The explored-final matched benchmark has been executed for all eleven models. The neural results are only 20-step smoke results: Direct LightGBM led (MAE 2.0392), DeepAR reached 3.0004 with substantial underforecasting, and N-HiTS reached 4.4305. These results diagnose undertraining; they are not research-trained neural comparisons. Six-origin statistical backtesting is complete. The ML pipeline passed a real-data one-origin matched smoke test; six-origin ML and DL execution remains compute-gated. Therefore the project does not yet claim a final eleven-model scientific winner.
