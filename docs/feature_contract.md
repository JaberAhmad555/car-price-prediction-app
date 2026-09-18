# Planned valuation feature contract — v0.1.0

This is a candidate Bangladesh feature vocabulary, not a trained model's input signature. **No listed field is yet guaranteed to exist with sufficient coverage in an eligible Bangladesh dataset.** Required serving inputs and supported ranges will be frozen only after data profiling and evaluation.

The normalized record schema may retain partial observations for review. Passing that schema does not make a record eligible for training or guarantee that the application can value it.

| Candidate field | Meaning / unit | Availability and handling |
| --- | --- | --- |
| `make` | Manufacturer | Intended core input; availability and normalization still unverified |
| `model` | Model within make | Intended core input; preserve source spelling |
| `variant` | Trim, package, edition | Optional candidate; null when unknown |
| `manufacture_year` | Calendar year manufactured | Candidate core input; do not conflate with model/registration year |
| `model_year` | Source-supported model year | Optional; do not invent it from another year |
| `registration_year` | Bangladesh registration year | Optional registration observation; no year-zero sentinel |
| `mileage_km` | Odometer distance in kilometres | Candidate core input; not fuel economy; zero only when observed as zero |
| `engine_cc` | Engine displacement in cubic centimetres | Candidate core input; null for unknown/not applicable, with raw context |
| `fuel_type` | Source-supported fuel classification | Vocabulary determined by data; unknown remains null |
| `hybrid_status` | Whether the vehicle is hybrid | Nullable boolean; unknown is not false |
| `transmission` | Transmission classification | Normalize only when supported; retain CVT/other detail where reliable |
| `body_type` | Body classification | Optional candidate |
| `colour` | Observed/reported colour | Optional; include in a model only if useful and reliable |
| `market_category` | Local Used / Foreign Used / Reconditioned / Brand New | Separate from physical damage and accident history; preserve original category |
| `previous_owners` | Number of previous owners | Optional, nonnegative integer; zero means known first ownership |
| `location` | Listing district/city | Candidate market context; no exact household location required |
| `registration_region` | Dhaka or other registration region | Optional reliable source/user field; plate number is not a model feature |
| `reported_condition` | Source-defined physical condition | Optional, evidence-qualified; never infer good condition from missingness |
| `accident_history` | Whether an accident is reported/verified | Optional nullable boolean qualified by evidence status; false is not inferred from silence |
| `auction_grade` | Grade with source grading context | Optional; grading systems cannot be blindly combined |
| `documentation_status` | Reported/verified documentation state | Optional; avoid collecting documents themselves without a justified purpose |
| `tax_token_valid_until` / `fitness_valid_until` | Source-supported validity dates | Optional only where permitted and reliable; not a claim of legal compliance |

Fields such as auction grade, condition, accident history, documentation, tax token, and fitness have particularly uncertain availability. Their presence in a record does not authorize production use. The future feature manifest must name selected fields, types, missing-value rules, categorical mappings, temporal rules, and supported segments.

## Labels and context are not customer price inputs

- `asking_price_bdt` is a supervised-training label, expressed in full BDT. It must not be a valuation predictor or required user input.
- **`Present Price` / `Present_Price` is not a required production feature and is deliberately absent from this schema.** Do not carry over the old feature mappings.
- Listing date, observation/collection date, provider, and raw provenance support lineage and evaluation. A provider ID is not automatically an appropriate predictive feature.
- Confirmed transaction prices belong to a separate future contract; they are not substitutes for asking prices.

## Normalization and leakage safeguards

Preserve original values in `provenance.raw_values` and identify the normalization version. Unknown optional fields can be omitted if unrecorded, or set to null if explicitly unknown; neither is a default value. Known zero odometer/owner counts are legitimate, but zero price or year is not a missing-value marker.

Fit transformations only on training folds, retain vehicle groups across repeated/cross-source listings, and respect observation time. No feature may contain the target, an unavailable future observation, or information unavailable at valuation time.

Photo-derived condition will eventually enter a separate evidence contract. Resolve it with other condition information before valuation and apply each condition signal once. Unknown or incomplete images must not become a favorable condition feature.
