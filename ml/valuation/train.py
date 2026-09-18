"""Reproducible historical Bangladesh experiments; no synthetic production data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .ingestion import ROOT

SEED = 42
SOURCE_ID = "mendeley-fmb4xmp4k5-v2"
CATEGORICAL = ["brand", "model", "transmission", "fuel_type", "body_type"]
NUMERIC = ["year", "mileage_km", "engine_cc"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Estimated Bangladesh Market Listing Value"


def metrics(actual, predicted):
    return {"mae_bdt": float(mean_absolute_error(actual, predicted)),
            "median_absolute_error_bdt": float(median_absolute_error(actual, predicted)),
            "rmse_bdt": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def fit_candidates(frame):
    """Group identical feature profiles before splitting; select using validation MAE."""
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).astype(str)
    if groups.nunique() < 30:
        raise ValueError("At least 30 distinct vehicle profiles are required for a three-way experiment")
    train_index, remaining = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=SEED).split(frame, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(frame.iloc[remaining], groups=groups.iloc[remaining]))
    validation_index, test_index = remaining[val_local], remaining[test_local]
    splits = {"train": train_index, "validation": validation_index, "test": test_index}
    if min(map(len, splits.values())) < 5:
        raise ValueError("Insufficient independent rows in an evaluation partition")
    candidates = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Random forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Histogram gradient boosting": HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15, l2_regularization=1, random_state=SEED),
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        preprocessing = ColumnTransformer([
            ("categories", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="unknown")),
                                     ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
            ("numbers", SimpleImputer(strategy="median"), NUMERIC),
        ])
        pipeline = Pipeline([("preprocess", preprocessing), ("regressor", estimator)])
        pipeline.fit(frame.iloc[train_index][FEATURES], frame.iloc[train_index].price_bdt)
        prediction = pipeline.predict(frame.iloc[validation_index][FEATURES])
        results[name] = metrics(frame.iloc[validation_index].price_bdt, prediction)
        fitted[name] = pipeline
    selected = min(results, key=lambda name: results[name]["mae_bdt"])
    winner = fitted[selected]
    residuals = frame.iloc[validation_index].price_bdt.to_numpy() - winner.predict(frame.iloc[validation_index][FEATURES])
    report = {
        "selected_model": selected, "selection_metric": "validation MAE in BDT",
        "validation_metrics": results,
        "test_metrics": metrics(frame.iloc[test_index].price_bdt, winner.predict(frame.iloc[test_index][FEATURES])),
        "residual_offsets_bdt": np.quantile(residuals, [.05, .95]).tolist(),
        "range_method": "Approximate 5th–95th percentile validation residual range; not a calibrated confidence interval or guaranteed coverage.",
        "split_rows": {key: len(value) for key, value in splits.items()},
        "split_indices": {key: value.tolist() for key, value in splits.items()},
        "split_method": "Seeded 60/20/20 group split by identical input profiles, not a temporal test. Model is fitted only on the training partition.",
        "random_seed": SEED,
    }
    return winner, report


def read_training_rows(run_dir):
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    receipt = run["receipt"]
    if (receipt.get("dataset_id") != SOURCE_ID or receipt.get("market") != "BD"
            or receipt.get("eligibility_status") != "APPROVED_FOR_EXPERIMENTATION"
            or not receipt.get("retrieved_on") or not receipt.get("file_sha256")):
        raise ValueError("An acquired, approved official Bangladesh dataset receipt is required")
    path = run_dir / "normalized.jsonl"
    rows, provenance = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["usable_for_historical_baseline"] or row["issues"]:
            continue
        vehicle, listing, observation = row["records"]
        if vehicle["market"] != "BD" or observation["currency"] != "BDT" or observation["price_type"] != "asking":
            raise ValueError("Unexpected market or price target")
        if vehicle["provenance"]["source_id"] != SOURCE_ID:
            raise ValueError("Unexpected source provenance")
        rows.append({"brand": vehicle["make"], "model": vehicle["model"], "year": vehicle["model_year"],
                     "mileage_km": observation["mileage_km"], "engine_cc": vehicle["engine_cc"],
                     "transmission": vehicle["transmission"], "fuel_type": vehicle["fuel_type"],
                     "body_type": vehicle["body_type"], "price_bdt": observation["asking_price_bdt"]})
        provenance.append(vehicle["provenance"]["source_record_ref"])
    if not rows:
        raise ValueError("No usable verified rows. Inspect units and quality flags before training.")
    frame = pd.DataFrame(rows).replace({None: np.nan})
    if frame[NUMERIC + ["price_bdt"]].isna().any().any() or (frame.price_bdt <= 0).any():
        raise ValueError("Incomplete numeric features or invalid prices")
    return frame, provenance, run, hashlib.sha256(path.read_bytes()).hexdigest()


def train(run_dir, output_dir):
    if output_dir.exists():
        raise ValueError("Model output directory already exists; use --output-dir for a new version")
    frame, provenance, run, normalized_hash = read_training_rows(run_dir)
    pipeline, report = fit_candidates(frame)
    options = {field: sorted(frame[field].dropna().unique().tolist()) for field in CATEGORICAL}
    options["models_by_brand"] = {brand: sorted(group.model.dropna().unique().tolist()) for brand, group in frame.groupby("brand")}
    options["bounds"] = {field: {"min": int(frame[field].min()), "max": int(frame[field].max())} for field in NUMERIC}
    insights = {
        "vehicle_count": len(frame), "average_price_bdt": int(round(frame.price_bdt.mean())),
        "median_price_bdt": int(frame.price_bdt.median()),
        "top_brands": [{"name": key, "count": int(count)} for key, count in frame.brand.value_counts().head(8).items()],
        "price_by_year": [{"name": str(int(key)), "price_bdt": int(group.price_bdt.mean()), "count": len(group)} for key, group in frame.groupby("year")],
        "price_by_fuel": [{"name": key, "price_bdt": int(group.price_bdt.mean()), "count": len(group)} for key, group in frame.groupby("fuel_type")],
    }
    metadata = {
        "artifact_schema_version": 1, "market": "BD", "currency": "BDT", "target": TARGET,
        "model_version": "bd-historical-" + normalized_hash[:12], "dataset_id": SOURCE_ID,
        "dataset_description": "Historical Bangladesh used-car listings; Mendeley V2 published January 2024. Not current 2026 prices.",
        "dataset_source": run["receipt"]["source_url"], "license": "CC-BY-4.0", "attribution": run["receipt"]["attribution"],
        "dataset_rows": len(frame), "raw_rows": run["actual_row_count"], "historical": True,
        "raw_sha256": run["file_sha256"], "normalized_sha256": normalized_hash,
        "trained_on": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "features": FEATURES, "options": options, "insights": insights, **report,
        "split_source_refs": {key: [provenance[index] for index in indexes] for key, indexes in report["split_indices"].items()},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "pipeline.joblib"
    joblib.dump(pipeline, artifact, compress=3)
    metadata["pipeline_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Existing verified ingestion run; otherwise detect under data/raw")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--raw", type=Path, help="Resolve multiple candidates by selecting a file inside --raw-dir")
    parser.add_argument("--sheet", help="XLSX worksheet when a workbook has several matching sheets")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    parser.add_argument("--price-unit", choices=["BDT", "lakh_BDT"], help="Only after reviewing bare source price units")
    parser.add_argument("--engine-unit", choices=["cc"], help="Only after reviewing bare source engine units")
    parser.add_argument("--unit-evidence", help="Document why any explicit bare-unit overrides are correct")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/valuation")
    args = parser.parse_args()
    if args.run_dir and (args.raw or args.sheet or args.price_unit or args.engine_unit or args.unit_evidence):
        parser.error("Do not combine an existing --run-dir with raw ingestion overrides")
    try:
        if args.run_dir is None:
            from .drop_in import prepare_run
            args.run_dir = prepare_run(args.raw_dir, raw=args.raw, sheet=args.sheet, encoding=args.encoding,
                                      price_unit=args.price_unit, engine_unit=args.engine_unit, unit_evidence=args.unit_evidence)
        metadata = train(args.run_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Training unavailable: {error}\nNo production model was created.\n")
    print(json.dumps({key: metadata[key] for key in ("selected_model", "raw_rows", "dataset_rows", "validation_metrics", "test_metrics")}, indent=2))
    print("Saved production bundle to " + str(args.output_dir.resolve()))
    print("Restart FastAPI, then reload the estimator to load dataset-derived Brand -> Model options.")


if __name__ == "__main__":
    main()
