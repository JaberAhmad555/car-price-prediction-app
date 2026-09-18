# Reviewed Bangladesh inference bundle

This directory deliberately ships the existing trained `pipeline.joblib` (about 1.16 MB) and its complete `manifest.json`. No training or raw dataset is needed at runtime. No retraining was performed for this release.

The manifest records source attribution, feature preprocessing, dependency versions, the artifact SHA-256, all candidate validation results, held-out metrics and residual quantiles. The loader verifies the checksum, Bangladesh/BDT contract and scikit-learn version before loading the operator-reviewed bundle.

Derived from **Car Dataset: Used cars data from Bikroy.com**, Mendeley Data V2, DOI [10.17632/fmb4xmp4k5.2](https://data.mendeley.com/datasets/fmb4xmp4k5/2), by Fahad Rahman Amik, Sifat Momen and Akash Lanard, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: normalization, review exclusions, training-only preprocessing and Random Forest fitting. No author endorsement is implied. Preserve this attribution with redistribution.

Treat joblib files as executable serialized artifacts: provision only reviewed project-generated bundles. `python -m backend.verify_release` checks inference before deployment. Other experimental artifacts remain ignored by Git.
