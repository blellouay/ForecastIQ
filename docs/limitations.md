# Limitations

- Recorded sales confound demand with stock availability and stockouts.
- M5 does not provide verified product launch dates for cold-start claims.
- Full-panel N-HiTS training includes all 30,490 M5 identities and therefore does not demonstrate unseen-product generalization.
- Identifier-free LightGBM artifacts are pilot-scale and lack a strict item-group holdout study.
- The five-series end-to-end panel is intentionally small.
- DeepAR lacks a verified deployable bundle and remains unavailable.
- N-HiTS negative outputs use a disclosed zero-clipping policy.
- Optional LLM explanations may be unavailable, rate-limited, or rejected; deterministic explanations remain available.
- Large data and model artifacts must be distributed outside ordinary Git.
