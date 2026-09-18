# Bangladesh data-source inventory — Phase 2A

Reviewed: **2026-09-18**. Historical MVP training has now been performed from the manually supplied official release. No marketplace collection, provider contact or live-feed integration has been performed.

Eligibility labels:

- `APPROVED_FOR_EXPERIMENTATION`: documented grant supports the stated limited experimental use; does not imply current-market fitness or production approval.
- `PERMISSION_REQUIRED`: a useful source route needs a provider/rights-holder agreement before acquisition.
- `NEEDS_REVIEW`: evidence is incomplete; no approved collection or experimentation use.
- `EXCLUDED`: unsuitable for the Bangladesh production-data purpose.

These inventory labels are separate from Phase 1's normalized `data_source.eligibility`. Experimental acceptance must not silently become `approved_for_valuation` or authorize production deployment.

## 1. Mendeley historical Bangladesh dataset, version 2

| Item | Verified evidence / decision |
| --- | --- |
| Source | Car Dataset: Used cars data from Bikroy.com; Mendeley Data |
| URL / DOI | https://data.mendeley.com/datasets/fmb4xmp4k5/2 ; `10.17632/fmb4xmp4k5.2` |
| Market | Bangladesh, according to the publisher's description |
| Description | Published historical second-hand car table described as collected from Bikroy |
| Record count | Publisher states 1,209; local CSV count independently confirms **1,209** |
| Fields | Publisher lists `car_name`, `brand`, `car_model`, `model_year`, `transmission`, `body_type`, `fuel_type`, `engine_capacity`, `kilometers_run`, `price`; local CSV headers match all ten fields |
| Dates | Published 2024-01-02, version 2. Original collection/listing dates not documented in accessible metadata. The user manually supplied car_dataset.csv; local receipt date and SHA-256 are recorded in the manifest. |
| License | Publisher explicitly declares CC BY 4.0 |
| Attribution | Credit Fahad Rahman Amik, Sifat Momen, Akash Lanard; include title, version, DOI, license link; identify changes and do not imply endorsement |
| Storage | Supported by CC BY's copying grant for the licensed material |
| ML experimentation | Historical experimentation accepted under CC BY's grant to adapt for any purpose; no claim of a separate publisher ML certification |
| Redistribution | Permitted by the declared license subject to its conditions and rights actually licensed. This repository will keep the full raw file ignored. |
| Automated collection | Official released-file download only was attempted; the host denied requests with HTTP 403. No authorization to collect live Bikroy listings is inferred. |
| Freshness | Historical release; not evidence of 2026 prices |
| Limitations | Local inspection found 1,209 rows, full BDT prices, cc engines and 1,141 usable rows after quality flags; see the measured quality report. Upstream collection permissions are not independently documented by the accessible record. |
| Eligibility | **APPROVED_FOR_EXPERIMENTATION**, limited to the published historical release under its declared license |

Evidence: [official metadata](https://data.mendeley.com/datasets/fmb4xmp4k5/2), [CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/), [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en). The license permits copying/adaptation and requires attribution and change notices; it supplies no warranty that all other rights are cleared. This limited decision is not permission to scrape the original marketplace.

See [manifest](../data/manifests/mendeley_fmb4xmp4k5_v2.json) for the machine-readable review, earlier failed retrieval attempts, actual file facts and the checksum-bound unit review.

## 2. Bikroy permissioned marketplace feed

| Item | Evidence / decision |
| --- | --- |
| Source / URL | Bikroy; https://bikroy.com/cars ; https://bikroy.com/rules.html |
| Market / description | Bangladesh vehicle advertisements; potential multi-seller current listing feed |
| Records / fields | No inventory collected or counted. Feed schema and available history are unverified; desired fields must be negotiated. |
| Dates / freshness | Potential current source; individual listing/collection dates and refresh cadence are not established |
| License / attribution | No reusable dataset license verified. Terms, provider and relevant content-owner permissions need review; attribution to be agreed. |
| Storage / ML / redistribution | Not approved; explicit permitted-use agreement required |
| Automated collection | Not approved. Terms section 7.16 requires user consent for copying/modifying/distributing user content; section 7.17 addresses collecting user information. No public API authorization was verified. |
| Limitations | Provider access, underlying content rights, history, source bias and freshness remain unresolved |
| Eligibility | **PERMISSION_REQUIRED** |

Only the published terms were researched; no listing scraper or collection was run. [Official terms](https://bikroy.com/rules.html)

## 3. D Autos showroom export / partnership candidate

| Item | Evidence / decision |
| --- | --- |
| Source / URL | D Autos; https://dautosbd.com/about |
| Market / description | Dhaka/Bangladesh dealer; potential permissioned stock-sheet export |
| Records / fields | No records obtained; count unknown. The provider describes price/auction sheets, repair-history information, registration/fitness/tax-token handling and inspections. An export schema is not verified. |
| Dates / freshness | Website reviewed 2026-09-18; inventory observation dates and delivery cadence unknown |
| License / attribution | No open data license or export agreement verified; attribution must be negotiated |
| Storage / ML / redistribution | Not approved; requires agreement defining each use |
| Automated collection | Not approved. Prefer a dealer-supplied export over website extraction. |
| Limitations | Claims are provider statements, not audited records; selected showroom inventory may not represent local-used cars or Bangladesh generally |
| Eligibility | **PERMISSION_REQUIRED** |

The [provider's description](https://dautosbd.com/about) makes this a candidate for discussion, not an established accessible dataset. Marketing stock/sales counters are not dataset statistics and are not used here.

## 4. Car Haat partnership candidate

| Item | Evidence / decision |
| --- | --- |
| Source / URL | Car Haat; https://www.carhaat.com.bd/contact |
| Market / description | Bangladesh/Dhaka vehicle selling, displaying and purchasing business described on its official contact page |
| Records / fields | Unknown; no export inspected or dataset received |
| Dates / freshness | Official search result reviewed 2026-09-18; direct page fetch timed out; record dates/cadence unknown |
| License / attribution | No dataset reuse license verified; requirements must be agreed |
| Storage / ML / redistribution / automation | None approved; a permissioned inventory export would require an agreement |
| Limitations | Practical willingness, field coverage, data ownership and usable history are unknown |
| Eligibility | **PERMISSION_REQUIRED** |

Reference: [official contact page](https://www.carhaat.com.bd/contact). No message was sent and no stock data was collected.

## 5. CarSell marketplace candidate

| Item | Evidence / decision |
| --- | --- |
| Source / URL | CarSell; https://www.carsell.com.bd/en/page/stay-safe |
| Market / description | Bangladesh vehicle marketplace candidate |
| Records / fields | Unknown for an obtainable dataset; no records collected or export inspected |
| Dates / freshness | Search result reviewed 2026-09-18; direct page fetch unavailable; collection/listing dates unknown |
| License / attribution | No data license or permission for training/redistribution established; safety/privacy information is not a data-use grant |
| Storage / ML / redistribution / automation | Unknown and unapproved |
| Limitations | Need applicable terms and explicit access/reuse evidence before considering acquisition |
| Eligibility | **NEEDS_REVIEW** |

## 6. Existing non-Bangladesh prototype artifacts

Reference: [historical preservation manifest](../legacy/manifest.json). Country/dataset details are incomplete; the prototype used non-Bangladesh price assumptions. Dataset size, fields beyond the saved prototype contract, original collection date and dataset license are not established. Historical artifact preservation is already authorized, but production training, derived Bangladesh predictions, redistribution of an unknown original dataset, and collection are not approved. Eligibility: **EXCLUDED** from Bangladesh production data and predictors.

## Current-source strategy, ranked separately by value and access

These are qualitative planning rankings, not measured dataset scores or commitments from providers.

| Priority | Route | Potential data value | Practical access | Next evidence needed |
| --- | --- | --- | --- | --- |
| 1 | Permissioned dealer/showroom export, e.g. D Autos or Car Haat | High for dated prices, vehicle specifications and possibly registration/inspection context | A bounded CSV/export pilot is a direct request; willingness and rights remain unconfirmed | Written storage/training/derived-output/display terms, sample schema, observation dates and provenance |
| 2 | Bikroy or another permissioned marketplace feed | High for broader seller/region coverage and repeated observations | Requires negotiated access; no public API availability assumed | Provider/content permissions, API/export contract, historical scope, refresh dates and deduplication keys |
| 3 | Manually curated records supplied with documented permission | Moderate for a reviewed pilot; limited scale and potential selection bias | Practical once individual suppliers consent; no copying workaround | Per-record provenance, permission and observation date |
| 4 | Additional licensed Bangladesh datasets | Depends on verified coverage and date; potentially useful for experiments | Search and license review needed; no current 2026 release verified in this phase | Actual files, collection dates, field coverage and source-rights evidence |

The recommended next current-data step is an agreed dealer export pilot, ideally with more than one supplier before wider market claims. No outreach, collection integration or automated feed is implemented here. Missing or uncertain rights never receive an approved label because access seems convenient.
