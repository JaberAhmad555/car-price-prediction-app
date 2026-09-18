# Historical Bangladesh ingestion — Phase 2A

The ingestion module prepares and validates local CSV/XLSX files without network collection. The training entry point now detects an official file placed under `data/raw/` and runs ingestion followed by the existing model experiments. See the [drop-in instructions](../../docs/data_drop_in.md). The supplied car_dataset.csv has been ingested and trained; measured results are in data/manifests/valuation_evaluation.json. Preserve unsupported file formats without renaming their extensions.

## Obtain and preserve the official file manually

1. Open https://data.mendeley.com/datasets/fmb4xmp4k5/2 and confirm DOI `10.17632/fmb4xmp4k5.2`, version 2.
2. Select **Download All**. The locally supplied table is car_dataset.csv.
3. Preserve the downloaded file/archive under `data/raw/`. Subdirectories are also searched. Keep its original name and bytes. If it is an archive, retain the archive and extract its CSV/XLSX alongside it; never overwrite either.
4. Record actual filenames, SHA-256 hashes, retrieval date and origin in the source manifest. Inspect actual headers/cells and any unit documentation before setting bare numeric unit options. An archive and its extracted table need separate hashes; the ingestion receipt identifies the table being processed.

Raw and generated datasets are ignored by Git. Do not force-add the full dataset. The source manifest records actual file facts and a checksum-bound unit profile. Automatic training reuses that profile only for identical raw bytes. The ingestion-run receipt contains resolved units and evidence.

## Run after the file and receipt are available

Install `backend/requirements-lock.txt` for ingestion and training, or `requirements-dev.txt` for ingestion only. The XLSX reader is pinned. For automatic detection and a complete run, execute `python -m ml.valuation.train` from the root. The following explicit ingestion command remains supported:

```sh
python -m ml.valuation.ingestion --raw "data/raw/car_dataset.csv" --receipt "PATH_TO_RESOLVED_RUN_RECEIPT.json" --output-dir data/processed/mendeley_fmb4xmp4k5_v2/run-001
```

The uppercase receipt path is a placeholder for the resolved receipt object in a prior run_manifest.json; export that receipt first or use the automatic training command instead. Choose a new output directory for each run. Existing output directories cause failure; raw files are only read. The receipt must have actual filename, checksum, retrieval date, approved experimental scope and any explicit unit evidence.

Outputs:

- `normalized.jsonl`: one envelope per source row, including three individually schema-validated records (vehicle, listing, listing observation), original values/provenance, parsing interpretation, issues and duplicate references. Every row is retained.
- `quality_report.json`: computed row counts, missingness, categories, rare groups, numeric summaries, duplicates and flag reasons.
- `run_manifest.json`: receipt, actual dimensions, raw checksum, encoding, parser/schema hashes, Python and validator versions.

Record IDs are deterministic file/row references, not invented marketplace listing IDs. Model year remains model year; manufacture/registration year, category, location, hybrid status and condition are not inferred. No fake condition or registration observations are generated when evidence is absent.

## Parsing and normalization

- Raw cell values remain unchanged under provenance. Text normalization applies Unicode NFKC, trims/collapses whitespace, and case-folds categorical comparison values; it does not merge synonyms or derive variants.
- Blank and explicit missing tokens become null, without filling zero/false. The original token is retained.
- Explicit BDT/৳/Tk prices parse as full taka. Explicit lakh/lac/লাখ multiplies by 100,000; crore/কোটি by 10,000,000. Decimal arithmetic must yield a whole BDT value; rounding is not silently applied.
- Bare numeric prices are **ambiguous** until `bare_price_unit` and its evidence are supplied. Neither value magnitude nor the desired output unit is sufficient proof of source units.
- Decimal points, valid South Asian/international comma grouping and Bengali digits are supported. Malformed commas, negatives, non-full-price financing text and foreign currencies are flagged, not guessed.
- `kilometers_run` identifies km for bare values; km suffixes are supported. Miles and abbreviated guesses such as `65k` are not silently converted.
- Engine cc suffixes and explicit litre suffixes are supported. Bare engine numbers remain ambiguous until a documented `cc` convention is configured. Do not silently interpret `1.5` as litres.

## Usability and reporting

"Usable" is a conservative historical-baseline candidate count: valid make/model/year/price/mileage/engine, no exclusion or review flags, and no later duplicate occurrence. It is not a training recommendation or current-market approval.

Exact duplicates compare all original fields; later occurrences are flagged as excluded but retained. A second check flags possible duplicates after normalization for review; failed numeric parses are not treated as equal just because both became null. Similar rows do not establish the same physical vehicle.

Review thresholds in the receipt are explicitly chosen screening heuristics, not measured market limits. Outliers stay in the output with reasons. Rare groups and strong category dominance are reported descriptively; this regression dataset has no fabricated classification accuracy.

Numeric summaries separately cover all parseable rows and usable rows. Invalid-price counts include unresolved units, financing and parse failures but exclude missing prices and review-only outliers, which have separate counts. Categorical counts and raw missingness cover all rows, including duplicates. Percentiles use linear interpolation. The source's published row count is compared to the observed count, never substituted for it.

The source schema/provenance contract must be validated with date-format checking. Cross-record identity, legal provenance, actual collection dates and source field semantics still require review; passing JSON Schema is not proof of those facts.
