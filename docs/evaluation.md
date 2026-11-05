# Evaluation

Validation MAE selects a model using three historical windows. Final-test MAE evaluates the already-selected forecast on d_1886–d_1913 and must not feed back into selection.

The reproducible five-series run is implemented by `python -m examples.verify_m5_nhits_e2e`. It covers all four demand types, requires N-HiTS to appear in every comparison, audits learned-model cutoffs, checks dates and identity, and writes `results/nhits_e2e/details.json` plus `summary.csv`.

This small integration panel verifies software and leakage controls. It is not a full M5 accuracy study or a claim that one model family is universally best.
