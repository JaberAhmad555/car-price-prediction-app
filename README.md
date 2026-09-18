# Bangladesh Used-Car Market Intelligence & AI Vehicle Valuation Platform

This project will estimate vehicle listing values and explain market patterns using eligible Bangladesh car-market data. Production prices will use **BDT / ৳ only**.

**The target is Estimated Bangladesh Market Listing Value, not confirmed transaction price.** Advertised asking prices and completed-sale prices are different observations; a removed listing does not prove a sale.

## Current status

Phase 1 establishes product contracts, a normalized record schema, repository boundaries, and validation tests. There is no production predictor, collected Bangladesh dataset, trained replacement model, frontend, or computer-vision implementation yet. No performance, dataset size, brand coverage, or live functionality is claimed.

The original non-Bangladesh Streamlit prototype and its artifacts are preserved only in [legacy/](legacy/README.md). They must never supply Bangladesh production predictions or act as a fallback. Future valuation models must be trained from scratch using eligible Bangladesh data.

## Planned capabilities

- Specification-based Bangladesh market listing valuation without photographs or a required "Present Price" input.
- Bangladesh market insights with documented coverage and observation periods.
- Optional AI photo condition inspection with visible findings, uncertainty, and coverage limitations.
- Later condition-aware valuation, only after the connection between condition and Bangladesh prices is validated.

Unavailable estimates, inspections, scores, and adjustments will remain unavailable. The product will not fabricate values to fill a screen or silently apply a condition discount.

## Contracts

- [Product scope](docs/product_scope.md)
- [Feature contract](docs/feature_contract.md)
- [Data source policy](docs/data_source_policy.md)
- [Normalized record schema](data/schemas/README.md)
- [Engineering rules](AGENTS.md)

## Repository structure

```text
frontend/                  Future Next.js / TypeScript / Tailwind interface
backend/                   Future FastAPI / Pydantic application
ml/
  valuation/               Bangladesh tabular ML
  computer_vision/         Optional visual inspection
  condition_adjustment/    Later validated valuation integration
  shared/                  Shared, versioned ML utilities
data/
  schemas/                 Normalized data contracts
  manifests/               Future dataset lineage and version metadata
  fixtures/                Small, explicitly labeled permitted test fixtures
docs/                      Product, data, and engineering documentation
tests/                     Phase 1 schema and isolation checks
legacy/                    Preserved historical prototype
.github/workflows/         Reserved for later CI
```

No framework scaffolding or database migrations are included in Phase 1. PostgreSQL/Supabase, Vercel, and suitable Python/ML hosting remain planned choices, not deployed services.

## Run the Phase 1 checks

Use Python 3.13 (verified locally with 3.13.5). The test dependencies are pinned in `requirements-dev.txt`; these are not the future production application's requirements.

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests use synthetic in-memory records only. They do not fetch data, load the legacy pickles, train models, or contact external services. The isolation checks reject legacy references in runtime/build files and copied historical artifacts. This is a regression guard, not a sandbox against arbitrary malicious code; future runtime artifact loaders must also enforce model provenance and compatibility.

`.env.example` contains empty placeholders only. Phase 1 tests need no secrets or external credentials. Never use `legacy/requirements.txt` as the production dependency list.

## Next phase

Phase 2 should start with source eligibility review, a versioned source manifest, and a Bangladesh data coverage plan. Acquisition and training require a subsequent instruction; they are not part of this change.
