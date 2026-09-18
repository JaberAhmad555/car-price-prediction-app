# Historical Bangladesh data-quality report

Computed from `data/raw/car_dataset.csv`; SHA-256 `cec972d4cf707424115d7d965ea3355d267c45f80543f82a1c3482c27e907e69`. Original bytes and a checksum-addressed snapshot remain untouched and ignored by Git.

[Official source and CC BY 4.0 attribution](https://data.mendeley.com/datasets/fmb4xmp4k5/2). The authors’ [paper, Table 1](https://www.mdpi.com/2078-2489/12/12/514) defines price in taka and engine capacity in cc. This review uses those explicit definitions, not price magnitude.

## Measured quality

| Measure | Result |
| --- | --- |
| Source rows / columns | 1209 / 10 |
| Usable historical rows | 1141 |
| Later exact duplicate rows | 1 |
| Invalid / missing prices | 0 / 0 |
| Distinct brands / model labels | 26 / 123 |
| Title/structured-year conflicts | 65 |
| Engine-capacity review flags | 2 |
| Missing body types | 18 |

All 1,209 price, mileage and engine values are bare integers. Prices are retained in full BDT (multiplier 1); mileage in km and engines in cc. No currency symbols, lakh strings or finance/deposit strings appear in the price column. All other source columns have zero missing values under the documented null parser.

One exact duplicate, 65 conflicting year rows and two engines below the 400 cc review threshold are excluded from this experiment, not deleted. Conflicting years are not guessed to be registration years. The 18 missing body types remain null in normalized records; the training-only categorical imputer uses an explicit unknown token. Mixed fuel categories and source spellings remain distinct; no unsupported equivalence is invented.

## Numeric distributions (all source rows)

| Field | Min | Median | Max |
| --- | --- | --- | --- |
| engine_capacity | 150 | 1,500 | 4,500 |
| kilometers_run | 6 | 62,000 | 900,000 |
| model_year | 1,983 | 2,011 | 2,021 |
| price | 120,000 | 1,520,000 | 16,000,000 |

Toyota accounts for 979/1,209 rows (80.98%); automatic transmissions account for 1,146/1,209 (94.79%). Rare models have insufficient evidence for a claim of reliable per-model accuracy. Low mileage observations and expensive outliers remain a limitation; they are not repaired based on intuition.

The complete machine-readable [quality report](../data/manifests/mendeley_quality_report.json) contains per-column missingness, category counts, rare groups, percentiles and row-level flag reasons. [Evaluation](../data/manifests/valuation_evaluation.json) records all candidate results and held-out metrics.

## Reproduction and limits

`python -m ml.valuation.train` uses the checksum-bound unit profile, produces a new ingestion run and requires a fresh artifact directory (use `--output-dir artifacts/<new-version>` for another run). The saved model uses 686 training rows; 227 validation rows select the winner and supply residual quantiles; 228 test rows supply final metrics. Preprocessing is fitted only on training rows.

Original listing/collection dates remain unknown. Publication on 2024-01-02 is not a listing date. There are no structured location, registration year, condition, accident, auction-grade, variant, owner-count or local/foreign/reconditioned-category fields. Hybrid appears in some fuel strings, but there is no dedicated hybrid-status field. The dataset cannot support current 2026 prices or verified transaction values.
