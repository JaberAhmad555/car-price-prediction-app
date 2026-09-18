# Permanent engineering rules

## Product and evidence

- Bangladesh production valuation data only. Train new valuation models from scratch on eligible Bangladesh records; all production valuation amounts use BDT / ৳.
- Predict **Estimated Bangladesh Market Listing Value** unless a separately defined target is supported by verified transaction data. Never equate a listing's disappearance with a sale.
- No production dependency on `legacy/`: no imports, loaders, copied artifacts, build inclusion, runtime requirements, or fallback predictions from the old project.
- Never fabricate model metrics, dataset statistics, coverage claims, condition scores, or predictions. UI mock/example prices require explicit labeling and must not masquerade as real outputs.
- `Present Price` must not be a required production feature. Avoid target leakage, future information, duplicate-vehicle leakage, and transformations fitted outside the training split.
- Unknown optional values remain unknown/null; missing condition or accident evidence is not a clean bill of health. Unsupported outputs remain explicitly unavailable.
- Photographs are optional. Do not connect computer-vision findings to monetary deductions without validated Bangladesh evidence. Avoid double-counting condition through both model features and an extra penalty.
- Future vision transfer-learning datasets may be evaluated only under documented compatible rights and must not substitute for Bangladesh validation. They do not provide production valuation labels.

## Data and reproducibility

- Do not automatically scrape websites. Review and document source terms, permitted acquisition, storage, training, derived-output use, attribution, and provenance before acquisition.
- Preserve source values, collection dates, normalization lineage, and verification status. Public visibility is not proof of reuse permission.
- Keep training reproducible with versioned datasets/models, schemas, preprocessing, configurations, split manifests, seeds, dependencies, code revisions, checksums, and recorded evaluation.
- Do not commit large datasets, generated model artifacts, user photographs, or sensitive records. Commit only small permitted fixtures labeled as synthetic/examples when applicable.
- User-image processing consent does not grant training consent. Training reuse requires separate explicit permission; private storage, deletion, and retention rules must be documented and tested.

## Implementation and verification

- Prefer readable, maintainable code and explicit contracts over unnecessary framework boilerplate.
- Keep training and serving transformations consistent. Do not silently change field meanings, currency units, or supported categories.
- Use environment variables for secrets. Keep real credentials out of code, examples, fixtures, logs, and Git.
- Run relevant tests before declaring a task complete. Phase 1 checks: `python -m unittest discover -s tests -v`.
- Extend isolation checks for new production/build surfaces. A static scan is not a substitute for a future trusted, versioned artifact-loading policy.
- Preserve original historical files and their integrity manifest. Do not fix or reuse the old model as part of production work.
- Report test failures, limitations, and unverified claims honestly. Follow the user's authorized phase boundaries; do not start collection, training, or deployment incidentally.
