# Product scope — contract v0.1.0

The product is the **Bangladesh Used-Car Market Intelligence & AI Vehicle Valuation Platform**. These contracts supersede the historical prototype. They describe planned behavior, not currently implemented capabilities.

## Production scope and target

- Production valuation training and evaluation use eligible Bangladesh car-market records. The existing non-Bangladesh model is historical only and must not be loaded or copied into production.
- Market is `BD`; currency is `BDT`. All normalized prices represent full taka amounts, never implicit lakh units or converted legacy predictions.
- The primary target is **Estimated Bangladesh Market Listing Value**: an estimate of a full advertised asking price for supported comparable vehicles, with the observation period and coverage documented.
- An asking/listing price is a seller's offer. A confirmed transaction price requires separate evidence of a completed sale. Do not merge the labels, infer a sale from delisting, or claim realized-sale accuracy from asking-price evaluation.
- Do not require customers to provide "Present Price" or the market price they want estimated.

## Planned capabilities

Specification-based valuation is independent of photo inspection. Market insights will describe supported Bangladesh segments with observation periods and sample counts derived from actual eligible data. Optional photo inspection may report visible exterior findings and image coverage; photographs cannot establish hidden mechanical health or accident-free history.

Candidate vehicle categories are Local Used, Foreign Used, Reconditioned, and Brand New. Supporting a category in a schema is not a claim that a trained model supports it. Reconditioned/Foreign Used describe market categories, not a damage severity or confirmed accident.

## Missing evidence and unsupported features

The eventual service must distinguish invalid input, unsupported segment, insufficient market data, unavailable model, inspection incomplete, and output unavailable. It must preserve usable specification-based valuation when photo inspection is absent or fails.

Unknown values remain null or explicitly unrecorded. No silent conversion to zero, false, undamaged, or a fabricated default is permitted. Return a reason for unavailable outputs; do not fill UI cards with invented prices, metrics, condition scores, or uncertainty percentages. Mock/example values are allowed only when explicitly labeled and separated from live outputs.

An interval must have measured held-out coverage. Model prediction uncertainty, damage-model confidence, and inspection coverage are different quantities. "No obvious exterior damage detected in submitted visible areas" is not equivalent to "damage-free"; unobserved areas remain unassessed.

## Condition and monetary adjustments

No automatic condition discount is permitted until an adjustment method is documented and validated using relevant Bangladesh evidence. In the absence of that evidence, keep inspection findings separate and mark the adjustment unavailable, not zero. An experimental label alone does not justify arbitrary monetary deductions.

Ordinary asking prices already reflect a mixture of vehicle conditions. Excluding a condition feature does not create a pristine-car reference price. A later reference-condition comparison must define its baseline and validate the final condition-aware estimate.

Use one resolved condition record with source, verification status, and uncertainty. Do not count the same damage across several images or through both a model feature and an additional penalty. Repair cost and market-value loss must not be assumed equal.

## Phase 1 boundary

This phase contains structure, documentation, a normalized record contract, and tests only. No collection, scraper, database deployment, model training, frontend, vision implementation, price-adjustment implementation, or full CI pipeline belongs here. Changes to supported features and claims require future data/model evidence and a versioned contract update.
