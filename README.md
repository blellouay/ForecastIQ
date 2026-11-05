# ForecastIQ
### Agentic AI for Adaptive Retail Demand Forecasting

ForecastIQ is a leakage-aware retail demand forecasting system that classifies demand behavior, determines which saved models are valid for a request, compares eligible candidates on identical historical windows, and produces a 28-day forecast through a LangGraph workflow. Different smooth, erratic, intermittent, and lumpy products can therefore use different forecasting methods.

## Core capabilities

- ADI/CV² demand classification: Smooth, Erratic, Intermittent, and Lumpy.
- Average, Naive, Seasonal Naive, SES, Croston, SBA, and TSB forecasts.
- Saved Direct and Hurdle LightGBM snapshots with frozen training mappings.
- Identifier-free pilot LightGBM models for history-bearing series.
- Full-panel N-HiTS snapshots at four historical origins.
- Separate inference and automatic-selection eligibility.
- Per-series rolling-origin validation and deterministic MAE routing.
- LangGraph orchestration with Python-owned numerical forecasts.
- Optional Groq explanations that cannot modify forecast values.
- Streamlit model availability, selection evidence, and downloads.

## Dataset

Research workflows use the [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset: 30,490 Walmart product-store sales series, calendar/event data, and weekly selling-price data. Integrated inference currently uses historical demand and calendar-derived fields. Selling prices are available in M5 but are not used because safe future price availability has not been established.

Raw M5 files are excluded from Git. Place them under `data/raw/` for full research runs.

## Architecture

```mermaid
flowchart LR
    A[Request and as-of sales history] --> B[Validate identity and cutoff]
    B --> C[Classify demand profile]
    C --> D[Assess model eligibility]
    D --> E[Three matched 28-day validation folds]
    E --> F[Per-series MAE comparison]
    F --> G[Select lowest eligible validation MAE]
    G --> H[Forecast from origin d_1885]
    H --> I[Consistency and nonnegative checks]
    I --> J[Forecast result]
    J --> K[Optional qualitative LLM explanation]
```

Numerical forecasts, selection, evidence, and totals remain Python-controlled. See [architecture.md](docs/architecture.md).

## Model comparison

| Family | Purpose | Automatic selection |
|---|---|---|
| Statistical | Baselines and intermittent-demand methods | Yes, with three matched folds |
| Direct LightGBM | Demand regression using a frozen mapping | Mapped identities and exact snapshots only |
| Hurdle LightGBM | Demand occurrence × positive quantity | Mapped identities and exact snapshots only |
| Generalizable LightGBM | Pilot features without product IDs | When every required snapshot validates |
| NHITS_small | Legacy saved checkpoint | Explicit inference only |
| NHITS_scalable | Full-panel snapshots at d_1801/1829/1857/1885 | Yes |
| Cold-start comparable | Transparent no-history fallback | No superiority claim |

DeepAR has training checkpoints but no verified deployable inference bundle, so it remains unavailable.

## Validation methodology

Selection compares the same product-store targets at origins **1801**, **1829**, and **1857**, with a 28-day horizon. The lowest mean per-series MAE wins. The final origin is **1885**, forecasting days 1886–1913. Final-test observations are opened only after selection.

Learned snapshots must declare a cutoff no later than their validation origin. N-HiTS uses exact-origin checkpoints and `nf.predict(df=...)` with truncated history. Negative N-HiTS outputs are clipped to zero before validation scoring and final output.

## Reproducible results

The real-M5 verification evaluated five mapped series across all four demand classes. N-HiTS appeared in every automatic comparison.

| Demand pattern | Selected model | Validation MAE | Final-test MAE |
|---|---|---:|---:|
| Lumpy | NHITS_scalable | 0.732706 | 1.142889 |
| Intermittent | NHITS_scalable | 0.242229 | 0.358748 |
| Smooth | TSB | 2.162270 | 3.452554 |
| Smooth | HurdleLightGBM | 2.677813 | 2.250806 |
| Erratic | NHITS_scalable | 1.875895 | 0.000908 |

These are per-series integration results, not a full-panel leaderboard. Five series cannot establish broad superiority. Details are under `results/nhits_e2e/`.

## Installation

Python 3.11 is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

This offline demonstration uses included synthetic history, no M5 files, credentials, or large checkpoints:

```bash
python -m examples.quickstart
```

Its output is demonstration evidence, not an M5 benchmark.

Full M5 verification:

```bash
python -m examples.verify_m5_nhits_e2e
```

This requires private M5 data and the artifacts in [models/README.md](models/README.md).

## Dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard shows diagnostics, inference/selection eligibility, validation MAE, forecasts, and CSV downloads. No screenshot is included because a publication-ready capture has not been reviewed.

## Optional Groq explanations

Never commit real keys. Configure the launching shell:

```powershell
$env:FORECASTIQ_LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:GROQ_MODEL = "openai/gpt-oss-20b"
python -m examples.run_step5
```

Use `FORECASTIQ_LLM_PROVIDER=offline` for a deterministic local explanation.

## Repository structure

```text
src/          forecasting, features, evaluation, and orchestration
dashboard/    Streamlit application
examples/     quick start and integration demonstrations
notebooks/    ordered research notebooks
models/       registries, metadata, and local artifacts
results/      reproducible summaries and manifests
docs/         architecture, methodology, evaluation, and limitations
tests/        unit and compatibility tests
scripts/      controlled artifact and research utilities
```

## Limitations

- M5 sales do not reveal stockouts, availability, or verified launch dates.
- Five-series verification is not evidence of global superiority.
- Generalizable LightGBM is a pilot, not a full item-group holdout study.
- Full M5 data and N-HiTS checkpoints are too large for ordinary Git.
- Full-panel N-HiTS does not demonstrate unseen-product generalization.
- Optional LLM commentary uses an external provider with an offline fallback.

See [limitations.md](docs/limitations.md).

## Future work

- Evaluate a preregistered, larger stratified panel.
- Add verified future price/promotion covariates.
- Export and validate a deployable DeepAR bundle.
- Add calibrated probabilistic intervals.
- Distribute optional artifacts through versioned releases.

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Model selection](docs/model_selection.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)

## License

No open-source license has been selected. Choose one before public publication.
