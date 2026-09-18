# Release deployment

The reviewed model and full metadata are committed under `artifacts/valuation/`. Raw data, uploads, caches, other models and local environments stay ignored. Do not run training during a release build.

## Render

- Service: Python web service; repository root (leave Root Directory empty).
- Python: `3.13.5`.
- Build: `pip install -r backend/requirements-lock.txt && python -m backend.verify_release`.
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Health path: `/health`.
- `CORS_ORIGINS`: exact public frontend origins, comma-separated; no wildcard. Add individual preview origins only when needed.
- `CARVALUE_MODEL_DIR=artifacts/valuation`.
- `REQUIRE_MODEL=1`: reject deployment startup if inference bundle validation fails.
- `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`.
- `PORT` is supplied by Render; no local drive path is required.

The loader checks the Bangladesh/BDT contract, scikit-learn version and pipeline checksum. The build check loads the shipped preprocessing/model and exercises health, metadata, options and a real-source specification prediction. A portability test also copies only backend + artifacts into a fresh temporary directory and runs inference there, without raw data or ML training modules.

## Vercel

- Root Directory: `frontend`.
- Framework: Next.js; Node.js 22.
- Install: `npm ci`; Build: `npm run build`; output: framework default.
- Required `NEXT_PUBLIC_API_URL=https://YOUR-API.onrender.com` (replace with the actual deployed origin).
- Set this value before building. Next.js embeds public variables at build time; changing it requires a new deployment.
- Production builds fail if the origin is missing or malformed. Vercel builds also reject local/non-HTTPS origins.
- Fonts are bundled locally and OpenGraph artwork is generated without remote assets.

Deploy the backend first, inspect `/health` for `model_ready: true`, set its URL on Vercel, then set the exact Vercel origin in backend CORS. Verify a prediction from the deployed browser and inspect its network/console errors. A free service may sleep, so allow for startup latency and verify the hosting tier meets the intended usage.

## Release integrity

`pipeline.joblib` and `manifest.json` are exact copies of the previously trained model bundle, not retrained artifacts. The manifest includes provenance, evaluation, split references, dependency versions and the pipeline checksum. See the bundle README for attribution. There is no data download, local filesystem lookup outside the repository, or training step at runtime.

Local production testing uses an explicitly supplied local `NEXT_PUBLIC_API_URL`; there is no localhost fallback in production. Only the development server has a local default. The frontend presents a retryable service error if the API cannot be reached.

Sources: [Render FastAPI deployment](https://render.com/docs/deploy-fastapi), [Next.js on Vercel](https://vercel.com/docs/frameworks/full-stack/nextjs), [Next.js environment variables](https://nextjs.org/docs/app/guides/environment-variables).

No hosting account, public URL or remote deployment has been created by this release preparation.
