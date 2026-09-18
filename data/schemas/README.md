# Normalized Bangladesh records — v0.1.0

`market_record.schema.json` is a self-contained JSON Schema Draft 2020-12 contract for **one record at a time**. All references are local; validation requires no network access. It defines six conceptual entities without creating tables, migrations, ingestion code, or an inference API.

| `entity_type` | Purpose / relationship |
| --- | --- |
| `data_source` | Source/version reference, terms review, acquisition method, scoped permissions |
| `vehicle` | Vehicle identity and specifications, with original source values |
| `listing` | A source advertisement referring to `vehicle_id`, category, date and location |
| `listing_observation` | A dated observation referring to `listing_id`, asking amount and odometer |
| `condition_observation` | Evidence-qualified physical condition/history referring to `vehicle_id` |
| `registration_observation` | Evidence-qualified Bangladesh registration/documentation referring to `vehicle_id` |

Every record has `schema_version`, `entity_type`, `id`, and `market: "BD"`. Each non-source record also has `provenance`, including source ID, original record reference, collection date/status, raw values, and normalization version. A source record's own reference, terms and review evidence establish its provenance; acquisition dates live on its collected records, not as an invented single date for the provider.

## Representation rules

- `asking_price_bdt` is a positive integer in **whole taka**, not lakhs or poisha. Null means no usable full asking price. Zero, negative, fractional, string, and boolean prices are rejected. Fractional taka are intentionally outside v0.1; any later support requires an explicit schema revision.
- `currency` is explicitly `BDT` and `price_type` is explicitly `asking`. Confirmed transaction amounts are not supported here and cannot be silently substituted; they need a separate future contract and evidence.
- Deposits, monthly payments, and amounts of unknown meaning must have a null normalized asking price. Keep their source amounts in `provenance.raw_values` for review.
- Required nullable fields must still be present. Optional fields may be omitted when unrecorded or set to null when explicitly unknown. The schema supplies no defaults and does not coerce or mutate values.
- Known zero mileage/previous-owner counts are valid; unknown is not zero. Year zero and engine displacement zero are not missing-value markers.
- Market category belongs to a listing. Physical condition and accident evidence belong to a condition observation. Neither implies the other.
- Dates use ISO calendar dates (`YYYY-MM-DD`). Enable a validator's format checker to reject impossible calendar dates. Unknown historical collection dates require `collection_date_status: "unknown"` and `collected_on: null`; publication/import dates are not substitutes. Known dates require status `known`.
- Retain the original fields in `raw_values` when normalizing; keep a durable snapshot reference where available. Do not put unnecessary personal identifiers or secrets into raw values.
- Unknown new fields are rejected outside `raw_values` to catch misspellings, unit changes, and accidental target leakage. Add new normalized fields through a documented schema revision.

## Validation and limits

The existing Phase 1 test dependencies provide the validator. From the repository root:

```python
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

schema = json.loads(Path("data/schemas/market_record.schema.json").read_text(encoding="utf-8"))
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema, format_checker=FormatChecker())
# validator.validate(record)  # record is a single JSON-compatible dictionary
```

Run `python -m unittest discover -s tests -v` for synthetic examples of all six entities and rejected inputs. No market records or source approvals are created by those tests.

Schema validity is not source approval, training suitability, or a promise of valuation support. Source permissions are declarations, not legal verification. Approval consistency is checked, but the evidence still needs human review under `docs/data_source_policy.md`.

Foreign-key existence, cross-record source agreement, unique identities, duplicate detection, field-level evidence review, plausible year relationships, observed-time ordering, source expiry, and training split eligibility are deliberately deferred to later ingestion/database validation. IDs are references here; this is not an ORM or database design. Only select documented, eligible fields into a future model; never pass the complete record (including its target) to training as features.
