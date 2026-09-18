# CarValue BD

### AI-Powered Bangladesh Vehicle Valuation Platform

CarValue BD estimates **Bangladesh used-car market listing values in BDT**, using a reproducible machine-learning pipeline and a full-stack valuation experience. Explore historical listing patterns, enter vehicle specifications, and inspect the evidence behind each estimate. Optional photo checks assess image quality; they do not detect damage or change prices.

**The target is Estimated Bangladesh Market Listing Value, not confirmed transaction price.** Historical data does not establish current 2026 market values.

## Live Demo

- **Website:** Not deployed yet — add the verified Vercel URL after deployment.
- **API:** Not deployed yet — add the verified Render URL after deployment.

Deployment configuration is prepared and locally verified. No public deployment is claimed.

## Features

- Responsive, three-step estimator with dataset-derived Brand → Model choices, validated specifications and animated BDT/lakh results.
- Real Random Forest predictions with an approximate range derived from validation residuals.
- Historical market insights: usable listing count, average/median prices, brand coverage, price by fuel and model year.
- Photo inspection **Beta**: multiple uploads, phone-camera capture, previews, removal, resolution, brightness and sharpness checks.
- Concise methodology, actual held-out metrics, provenance and explicit limitations.
- Accessible labels, keyboard focus, mobile navigation, reduced-motion support, error states and loading feedback.
- Typed API, configurable CORS, checksum-verified model loading and automated release checks.

## Architecture

```text
Next.js Frontend (Vercel)
        ↓ JSON / HTTPS
FastAPI API (Render)
        ↓ validated vehicle specifications
Bangladesh ML Pipeline (saved preprocessing)
        ↓
Random Forest Model → BDT estimate + approximate residual range

Vehicle Photos
        ↓
Image Quality Analysis (in memory)
        ↓
Future Damage Detection Module [not implemented]
```

```text
frontend/              Next.js, TypeScript, Tailwind, bundled typography
backend/               FastAPI, model loader, release verification, photo checks
artifacts/valuation/   Reviewed inference pipeline + complete model manifest
ml/valuation/          Reproducible ingestion and training tools
 data/schemas/         Normalized data contracts
 data/manifests/       Attribution, source review, quality and evaluation reports
tests/                 Parsing, provenance, isolation, API and portability checks
docs/                  Product contracts, data policies and deployment instructions
legacy/                Preserved original prototype; never a production dependency
```

## Machine Learning

| Property | Actual result |
| --- | --- |
| Raw records | 1,209 |
| Usable records | **1,141** |
| Model | **RandomForestRegressor** |
| Trees | **160** |
| MAE | **৳234,447.54** |
| RMSE | **৳399,114.69** |
| R² | **0.918655** |
| Median absolute error | ৳123,324.96 |

Metrics come from this project's **held-out test split**, not the source paper. The split contains 686 training, 227 validation and 228 test rows, with seed 42 and identical input profiles grouped together. Preprocessing is fitted only on training rows and saved with the regressor. Real source vehicle identities are unavailable, so grouping does not guarantee that every physical vehicle is unique across partitions.

| Candidate | Validation MAE (BDT) |
| --- | ---: |
| Median baseline | 1,159,933.92 |
| Random forest — selected | **432,040.20** |
| Histogram gradient boosting | 459,731.43 |

Validation MAE selects the model; the test set does not. The model remains fitted on the training partition. The validation/test error difference highlights sampling uncertainty and expensive outliers; R² is not a percentage of predictions that are correct.

Inputs: brand, model, model year, mileage in km, transmission, fuel type, body type and engine capacity in cc. Customers never need to supply “Present Price.” Normalization preserves source values, nulls, hashes and exclusion reasons. One duplicate, two suspicious engines and 65 conflicting title/year rows are retained for review and excluded from this experiment.

The displayed range uses the 5th and 95th percentiles of validation residuals, includes the estimate and clips its lower bound at zero. It is **not a calibrated confidence interval**; the same validation partition selected the model. [Complete evaluation](data/manifests/valuation_evaluation.json) · [Quality report](docs/data_quality_report.md).

## Dataset

**Car Dataset: Used cars data from Bikroy.com** — Mendeley Data, **Version 2**, DOI [10.17632/fmb4xmp4k5.2](https://data.mendeley.com/datasets/fmb4xmp4k5/2), published January 2, 2024.

Credit: **Fahad Rahman Amik, Sifat Momen and Akash Lanard**. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Project modifications include normalization, explicit review exclusions and model fitting; no endorsement is implied.

This is a **historical Bangladesh dataset**, not the whole Bangladesh market. Original collection and listing dates remain unknown. The authors’ [Table 1](https://www.mdpi.com/2078-2489/12/12/514) defines full taka prices and engine capacity in cc. No lakh multiplier is applied to the source price column.

The raw CSV is excluded from Git and unnecessary for inference. The small reviewed model bundle is shipped explicitly so a fresh clone can predict without downloading data or training. The dataset license does not authorize scraping live marketplace pages.

## Technology Stack

Next.js · TypeScript · Tailwind CSS · FastAPI · Python · scikit-learn · pandas · OpenCV/Pillow image-quality processing · GitHub Actions. Manrope is bundled through Fontsource, avoiding a build-time font service dependency.

## Local Development

Use **Python 3.13.5** and **Node.js 22**. From the repository root, in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements-lock.txt
npm --prefix frontend ci
.venv/Scripts/python -m backend.verify_release
```

Terminal 1:

```powershell
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
npm --prefix frontend run dev
```

Open `http://localhost:3000`; API documentation is at `http://127.0.0.1:8000/docs`. On macOS/Linux use `.venv/bin/python` instead.

Development defaults to the local API. **Production builds require `NEXT_PUBLIC_API_URL` explicitly**, because public Next.js variables are embedded at build time. For a local production check:

```powershell
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix frontend run start
.venv/Scripts/python -m unittest discover -s tests -v
```

Backend environment: `CORS_ORIGINS` (comma-separated exact frontend origins), `CARVALUE_MODEL_DIR` (default `artifacts/valuation`), `REQUIRE_MODEL=1` (fail startup if the model cannot load). Backend environment variables are supplied by the shell/host; `.env` is not loaded implicitly. Keep secrets out of public frontend variables.

For optional future retraining, follow the [drop-in workflow](docs/data_drop_in.md); training is not part of application startup or deployment. The original raw file and earlier experiment runs must remain intact.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Process health and model readiness |
| GET | `/api/metadata` | Model, historical-data scope and actual metrics |
| GET | `/api/options` | Supported categories, brand/model pairs and numeric bounds |
| POST | `/api/predict` | Full integer BDT estimate, lakh equivalent and approximate range |
| GET | `/api/insights` | Calculated historical dataset summaries |
| POST | `/api/inspect` | In-memory quality checks for a JPEG, PNG or WebP request body |

Example request (specifications from an actual source record; response is calculated):

```json
{
  "brand": "toyota", "model": "axio", "year": 2016,
  "mileage_km": 96000, "engine_cc": 1500,
  "transmission": "automatic", "fuel_type": "cng, hybrid", "body_type": "saloon"
}
```

Unsupported inputs return `422`; unavailable models return `503`. Images are limited to 8 MB / 16 megapixels and processed in memory. Image checks never apply a monetary deduction.

## Deployment

**Render:** repository root; build `pip install -r backend/requirements-lock.txt && python -m backend.verify_release`; start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Set `CORS_ORIGINS` to your actual Vercel/custom-domain origins. The Blueprint pins Python and requires a ready model.

**Vercel:** root directory `frontend`, Next.js framework, Node.js 22. Set `NEXT_PUBLIC_API_URL` to the public HTTPS Render origin **before building**. Redeploy after changing it.

The tracked model is about 1.16 MB and its full manifest about 96 KB. The runtime has no raw-data, training-module, local-drive or historical-prototype dependency. [Deployment guide and release verification](docs/deployment.md).

## Limitations

- Historical asking prices cannot guarantee today's listing value or an actual sale price.
- Toyota and automatic vehicles dominate the sample; rare brands/models have limited evidence.
- No structured registration, condition, accident, ownership, location or auction-grade fields support valuation adjustments.
- No temporal evaluation or verified current-market coverage; residual ranges are approximate.
- Photo inspection assesses image quality only. Visible-damage recognition, mechanical inspection and condition-price adjustments are not implemented.
- Uploaded images are not stored by this application or used for training. Avoid unnecessary faces, plates or documents in uploads.
- This stateless MVP has no accounts or stored valuation history. Public hosting, cold-start behavior and service capacity still require verification after deployment.

## Roadmap

- Larger, current Bangladesh datasets with documented rights and observation dates.
- Permissioned dealer integrations and listing-history exports.
- Proper visible-damage detection evaluated on licensed datasets.
- Condition-aware valuation supported by repair/market evidence, without double-counting damage.
- Model monitoring, drift evaluation and versioned retraining.

## Project / Portfolio

Built by [Jaber Ahmad](https://github.com/JaberAhmad555). This portfolio demonstrates reproducible tabular ML, evidence-based product boundaries, API engineering and a responsive full-stack interface. The original prototype remains recoverable under `legacy/`; its non-Bangladesh model is never used for this application.
