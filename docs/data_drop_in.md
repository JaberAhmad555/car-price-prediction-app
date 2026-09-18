# Enable real Bangladesh predictions

Download **Car Dataset: Used cars data from Bikroy.com**, Mendeley Data **version 2**, DOI **10.17632/fmb4xmp4k5.2**, from the [official release](https://data.mendeley.com/datasets/fmb4xmp4k5/2).

Place the unmodified source table in **`data/raw/`**, relative to this repository. Keep its original filename. Accepted names are **any `*.csv` or `*.xlsx`**, case-insensitive; no specific basename is required. If the official Download All button returns a ZIP, preserve it and extract the contained CSV/XLSX into `data/raw/` or a subdirectory. ZIP files are not themselves training inputs. `.xls`, renamed non-CSV files, and formula-dependent spreadsheets are unsupported.

The supplied and inspected release is `car_dataset.csv`, with 1,209 rows and ten columns. Detection checks actual headers: `car_name`, `brand`, `car_model`, `model_year`, `transmission`, `body_type`, `fuel_type`, `engine_capacity`, `kilometers_run`, `price`. Case/space formatting is normalized. Other columns are preserved. The matching worksheet is selected automatically when unique. Matching columns identify a candidate, not its legal provenance: only place the official licensed V2 release here, and review the supplied file before training.

## One training command

From the repository root in PowerShell:

```powershell
.venv/Scripts/python -m pip install -r backend/requirements-lock.txt
.venv/Scripts/python -m ml.valuation.train
```

The command automatically detects the table, preserves a checksum-addressed copy under `data/raw/snapshots/`, records a local receipt in the ingestion run, validates and normalizes rows, compares the existing three candidate models, and saves `artifacts/valuation/pipeline.joblib` plus `manifest.json`. Neither source bytes nor an existing production bundle are overwritten. The local receipt records the date received by this workflow; it does not invent the original web download date or collection/listing dates.

The manifest records **actual raw and usable rows, validation MAE/RMSE/R²/median absolute error, selected-model held-out test metrics**, split membership, source hashes and real Brand → Model options. No file means no training, no model and no price output.

If several distinct matching files exist, select one with `--raw "data/raw/<original filename>.csv"`. For several matching Excel sheets, add `--sheet "<actual sheet name>"`. Byte-identical source/snapshot copies are treated as one candidate. CSV uses UTF-8 by default; use `--encoding` only when the original encoding is known.

## Units are never guessed

Explicit currency/cc/litre units are parsed automatically. The reviewed `car_dataset.csv` checksum has a recorded unit profile: bare prices are full taka (BDT), engines are cc, and mileage is km. The authors’ [Table 1](https://www.mdpi.com/2078-2489/12/12/514) supplies the unit definitions. Automatic reuse requires identical file bytes. For other files with undocumented bare units, the workflow stops and lists unresolved fields. After checking the actual source documentation, supply only the needed options:

```powershell
.venv/Scripts/python -m ml.valuation.train --price-unit BDT --engine-unit cc --unit-evidence "Describe the actual source evidence confirming these units"
```

Use `--price-unit lakh_BDT` only if evidence establishes lakh-based source values. The text above is an evidence placeholder, not an assertion that Mendeley uses these units. Magnitude alone is not proof. Missing, negative, suspicious and duplicate rows retain explicit flags; they are not silently rewritten into valid labels.

## Activate the resulting bundle

Restart FastAPI, then reload the frontend estimator:

```powershell
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`GET /health` must report `model_ready: true`. `GET /api/options` supplies actual dataset brands and model pairs; `POST /api/predict` returns full integer BDT values. The frontend uses **Estimated Bangladesh Market Listing Value**, comma-grouped BDT and a two-decimal lakh equivalent. It remains explicitly historical, not a current transaction-price prediction.

If `artifacts/valuation` already exists, use a fresh versioned `--output-dir artifacts/<new-name>` and set `CARVALUE_MODEL_DIR` to that directory before restarting. Old bundles and raw records remain recoverable.
