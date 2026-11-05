# ForecastIQ model artifacts

Registries and lightweight metadata may be committed; large weights should not be.

| Family | Location | Historical origins |
|---|---|---|
| Direct/Hurdle LightGBM | `models/lightgbm/`, registry `models/registry.json` | 1801, 1829, 1857, 1885 |
| Generalizable LightGBM pilot | `models/generalizable_lightgbm/` | 1801, 1829, 1857, 1885 |
| NHITS_small | `models/neuralforecast/nhits_small/` | trained through 1857 |
| NHITS_scalable | `models/nhits_full_m5/full_all/origin_*` | 1801, 1829, 1857, 1885 |

Each learned snapshot must preserve its alias, feature schema or NeuralForecast configuration, training cutoff, horizon, and provenance. The full N-HiTS bundles require `configuration.json`, `configuration.safetensors`, `dataset.json`, `dataset.safetensors`, the model safetensor, and provenance JSON.

Load N-HiTS only through `src.agents.nhits_inference`, which validates provenance, forces CPU inference when requested, supplies truncated history, and checks dates and values. Load trusted local joblib artifacts only through the registered LightGBM adapters.

No external artifact URL is currently published. Obtain the original private artifacts from the project owner and place them at the documented paths. Never substitute similarly named weights.
