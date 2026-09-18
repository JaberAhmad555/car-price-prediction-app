"""Prepare a locally supplied official Mendeley release; never download or invent data."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from .ingestion import ROOT, digest, ingest, load_source
from .parsing import parse_engine_with_unit, parse_price

OFFICIAL_URL = "https://data.mendeley.com/datasets/fmb4xmp4k5/2"
DOI = "10.17632/fmb4xmp4k5.2"


def discover_dataset(raw_dir, *, raw=None, sheet=None, encoding="utf-8-sig"):
    """Find by actual columns, not filename; reject ambiguous distinct datasets."""
    raw_dir = Path(raw_dir).resolve()
    paths = [Path(raw).resolve()] if raw else sorted(
        (path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"}),
        key=lambda path: (len(path.parts), str(path)),
    )
    matches, errors = {}, []
    for path in paths:
        if not path.resolve().is_relative_to(raw_dir):
            raise ValueError("Supply the official file inside " + str(raw_dir))
        try:
            content, headers, rows, worksheet = load_source(path, sheet=sheet, encoding=encoding)
            matches.setdefault((digest(content), worksheet), (path, content, headers, rows, worksheet))
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            errors.append(f"{path.name}: {error}")
    if len(matches) > 1:
        choices = "\n".join(str(value[0]) for value in matches.values())
        raise ValueError("Multiple distinct matching files. Select the official release using --raw:\n" + choices)
    if not matches:
        detail = "\n" + "\n".join(errors) if errors else ""
        raise ValueError(
            "DATASET FOUND: NO\nDownload: Car Dataset: Used cars data from Bikroy.com, Mendeley V2\n"
            f"DOI: {DOI}\nOfficial source: {OFFICIAL_URL}\n"
            f"Place the unmodified file in: {raw_dir}\n"
            "Accepted filenames: any original filename ending in .csv or .xlsx (case-insensitive).\n"
            "If Download All returns a ZIP, keep the ZIP and extract its CSV/XLSX into this folder.\n"
            "The table must contain the ten published source columns." + detail
        )
    return next(iter(matches.values()))


def prepare_run(raw_dir, *, raw=None, sheet=None, encoding="utf-8-sig", price_unit=None,
                engine_unit=None, unit_evidence=None, processed_dir=None):
    path, content, headers, rows, worksheet = discover_dataset(raw_dir, raw=raw, sheet=sheet, encoding=encoding)
    if not rows:
        raise ValueError("The matching source table is empty")
    checksum = digest(content)
    receipt = deepcopy(json.loads((ROOT / "data/manifests/mendeley_fmb4xmp4k5_v2.json").read_text(encoding="utf-8")))
    reviewed = receipt.get("verified_unit_profile", {})
    if checksum == reviewed.get("file_sha256") and not (price_unit or engine_unit or unit_evidence):
        price_unit = reviewed["price_unit"]
        engine_unit = reviewed["engine_unit"]
        unit_evidence = reviewed["evidence"]
    if (price_unit or engine_unit) and not (unit_evidence and unit_evidence.strip()):
        raise ValueError("Explicit bare units require --unit-evidence describing source documentation or manual verification")
    ambiguous = Counter()
    for row in rows:
        if parse_price(row["mapped"]["price"], bare_unit=price_unit).issue == "ambiguous_price_unit":
            ambiguous["price"] += 1
        if parse_engine_with_unit(row["mapped"]["engine_capacity"], bare_unit=engine_unit).issue == "ambiguous_engine_unit":
            ambiguous["engine_capacity"] += 1
    if ambiguous:
        raise ValueError(
            f"Found {path}, but source units need review: {dict(ambiguous)}. "
            "After verifying the official source, supply --price-unit BDT (or lakh_BDT), "
            "--engine-unit cc as needed, and --unit-evidence. Magnitude alone is not unit evidence."
        )
    now = datetime.now(timezone.utc)
    receipt.update({
        "original_filename": path.name, "file_sha256": checksum,
        "retrieved_on": now.date().isoformat(), "retrieval_date_basis": "Date received by this local workflow; original web download date is unknown",
        "raw_status": "LOCAL_RELEASE_DECLARED_BY_OPERATOR",
        "source_verification": "Operator supplies the official V2 release; matching headers do not independently prove provenance",
        "actual_row_count": len(rows), "actual_column_names": headers, "worksheet": worksheet,
        "intended_allowed_use": "Historical Bangladesh ingestion and authorized MVP experiments under the declared CC BY 4.0 license",
    })
    receipt["normalization"].update({
        "price_format_status": "EXPLICIT_CELL_UNITS_OR_OPERATOR_DOCUMENTED_UNITS",
        "bare_price_unit": price_unit, "bare_price_unit_evidence": unit_evidence if price_unit else None,
        "bare_engine_unit": engine_unit, "bare_engine_unit_evidence": unit_evidence if engine_unit else None,
    })
    # Preserve a checksum-addressed copy before writing derived records. Never overwrite it.
    snapshot_dir = Path(raw_dir) / "snapshots" / checksum
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / path.name
    if snapshot.exists():
        if digest(snapshot.read_bytes()) != checksum:
            raise ValueError("The preserved snapshot has changed; resolve it before training")
    else:
        with snapshot.open("xb") as stream:
            stream.write(content)
    run_dir = Path(processed_dir or ROOT / "data/processed") / ("mendeley-v2-" + checksum[:12] + "-" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    report = ingest(snapshot, receipt, run_dir, encoding=encoding, sheet=worksheet)
    print(json.dumps({"dataset_file": str(path), "preserved_snapshot": str(snapshot), "worksheet": worksheet,
                      "raw_rows": report["raw_row_count"], "usable_rows": report["usable_row_count"],
                      "columns": headers, "ingestion_run": str(run_dir)}, ensure_ascii=False, indent=2))
    return run_dir
