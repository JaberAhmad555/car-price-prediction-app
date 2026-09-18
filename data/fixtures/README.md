# Test fixtures

`ingestion_synthetic.json` contains nine deliberately invented rows for local parser and ingestion tests. The records were authored for this project and contain no third-party data or personal information. They may be committed under the project's test-fixture policy. They are **not** a sample of the Mendeley release and must never be used for model training or reported as Bangladesh market statistics.

Cases cover an exact duplicate, formatting-equivalent possible duplicate, explicit BDT/lakh units, Bengali digits, missing fields, negative values, a down payment, review outliers and ambiguous bare numbers. Tests materialize temporary CSV files from the JSON; the source fixture is unchanged. Blank CSV cells are preserved as blank strings in raw provenance and normalized to null.

Actual source fixtures require a separate rights and privacy review after the official file is obtained. None have been created in Phase 2A.
