# Current Bangladesh market gap — Phase 2A

The reviewed Mendeley release is historical: its official publication date is 2024-01-02. It cannot, by itself, substantiate current 2026 Bangladesh asking prices. The supplied car_dataset.csv has now been inspected: 1,209 rows and ten columns.

## What is known versus unverified

The published description lists make/brand, model/name, model year, transmission, body type, fuel, engine capacity, kilometres run, and price. These ten headers and their cell values were inspected for this MVP.

| Required current-market evidence | Metadata observation | Actual-file conclusion |
| --- | --- | --- |
| Current 2026 asking prices | Release published in January 2024 | No evidence of 2026 observations; prices are bare full-BDT integers |
| Listing and collection dates | Not in the listed columns; original collection date not documented | No dedicated date columns; collection dates remain undocumented |
| Location | Not named in published columns | No structured location field; no verified extraction from free text |
| Bangladesh registration year/region | Not named | No dedicated structured field |
| Local Used / Foreign Used / Reconditioned / Brand New | Not named | No dedicated structured field |
| Hybrid status | Fuel type is named; separate hybrid flag is not | No dedicated hybrid flag; some fuel strings explicitly contain Hybrid |
| Variant/package | Car/model names are listed; no separate variant field named | No dedicated variant field; titles sometimes contain package text |
| Ownership history | Not named | No dedicated structured field |
| Accident history and physical condition | Not named | No dedicated structured field |
| Auction grade | Not named | No dedicated structured field |
| Source listing IDs and per-record timestamps | Not named | No original ID or timestamp columns; generated file/row references are not marketplace IDs |
| Confirmed transaction prices | Dataset describes marketplace cars/prices | No transaction evidence established; do not relabel asking prices as sales |

Missing structured fields must not be filled from assumptions. Free-text fragments are not verified condition, registration or ownership records. Publication, retrieval and listing dates remain distinct. See the measured data-quality report for limitations including 65 title/year conflicts.

## Additional evidence needed for a current product

Obtain permissioned, dated Bangladesh observations with documented price basis and update cadence. Prioritize make/model/package, manufacture and registration dates, odometer, engine, fuel/hybrid, transmission, body type, market category and district/city. Record unknowns and verification status rather than inventing condition, auction or accident information.

Maintain stable source identities, repeated listing observations and cross-source deduplication evidence. Seek more than one seller/source and appropriate geographic and vehicle coverage before making broad market claims. Measure actual sample counts and missingness once acquired; no minimum dataset size is invented here.

Hold out recent observations and vehicle groups for later evaluation. An undated historical sample cannot justify a future-price test. Current-market price drift, seller selection, duplicated ads, asking-versus-sale gaps and changing category coverage must be measured with new evidence.

## Recommended acquisition route

Start with a written, permissioned dealer/showroom export pilot specifying storage, experimentation, derived predictions, public display, attribution and retention. Include observation dates and field definitions in the agreement. D Autos and Car Haat are discussion candidates, not approved suppliers. A permissioned marketplace feed may later improve coverage, but no API or automated collection right has been established.

See the [source inventory and rankings](data_sources.md). No provider has been contacted, no current-market collector has been built, and only the historical Bangladesh baseline model has been trained.
