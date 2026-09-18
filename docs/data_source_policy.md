# Bangladesh data source eligibility policy — v0.1.0

**Publicly visible data must NOT automatically be assumed to be legally reusable for automated scraping or ML training.** No scraper or acquisition integration is implemented in Phase 1.

## Source register

Before accepting records, retain the following for each source and version:

| Record | Required information |
| --- | --- |
| Identity | Source/provider, URL or durable reference, Bangladesh geographic scope |
| Acquisition | Published dataset, provider export, manual curation, API, or permitted automated collection |
| Terms | License/terms/agreement reference and version, review date, supporting evidence |
| Use permissions | Separate decisions for storage, automated collection, ML training, derived predictions, public display, and redistribution |
| Attribution | Exact required credit or explicit confirmation that no attribution is required |
| Restrictions | Retention, rate/access limits, expiry, withdrawal obligations, and other conditions |
| Provenance | Collection date per acquisition/record, source record ID/reference, import/snapshot reference, raw values, normalization version |

Permission values are `permitted`, `prohibited`, or `unknown`. Unknown is not permission. An uploader's license declaration is evidence to review, not an automatic resolution of upstream rights. Manual copying is not a workaround for source restrictions. Access/robots rules alone do not establish training or redistribution rights.

## Eligibility decisions

`pending_review` is the initial state until a reviewed decision exists; `rejected` records are excluded. `approved_for_valuation` means storage, training, and derived predictions have been reviewed and permitted for the specified Bangladesh source/version. Automated API or website acquisition additionally needs permitted automated collection. Public display and redistribution remain separate even for an approved training source.

The JSON Schema enforces basic consistency of these declarations. It cannot determine whether a legal conclusion is correct. The source register must retain evidence, a review date, restrictions, and attribution instructions; runtime ingestion/training will later enforce this policy against each manifest and intended use.

When terms change or access is withdrawn, stop affected new acquisition, record the change, and evaluate stored datasets and model releases against the applicable agreement. Keep lineage sufficient to locate affected records and releases.

## Record-level provenance

Every normalized vehicle, listing, listing observation, condition observation, and registration observation needs provenance. Preserve source values whenever normalizing language, units, categories, or dates. Track data collection separately from listing publication and database import.

New acquisitions must capture their collection date. An older published dataset may have an unknown original collection date: keep null, identify it as unknown, and retain dataset DOI/version and row reference. Never substitute publication/import date or invent per-listing timestamps. Such data cannot support claims of time-based validation without additional evidence.

Preserve source market labels and verification status. Reject or quarantine non-Bangladesh valuation records, invalid normalized prices, and non-full-price advertisements rather than silently treating deposits/installments as the target. A source may be approved while particular records remain unsuitable for training.

## Dataset/model release requirements

Future manifests must list eligible source versions, intended uses, attribution, record/snapshot checksums, collection period and missingness, normalization version, deduplication strategy, split definitions, and applicable restrictions. No source is approved and no dataset statistics are asserted by these Phase 1 documents.

The current schema covers Bangladesh tabular records. Future computer-vision data requires separate review for images, annotations, software, and weights, plus Bangladesh evaluation. User uploads remain private and excluded from training unless separately and explicitly permitted; inference consent is insufficient.
