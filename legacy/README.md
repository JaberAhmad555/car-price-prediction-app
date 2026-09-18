# Original prototype — historical reference only

This is the original Streamlit car-price prototype based on non-Bangladesh data. It is preserved only for project history and **must not be used for Bangladesh production predictions**, including fallback predictions or copied model weights.

The original app, model, SHAP explainer, and requirements were moved here without edits. The precise original dataset and training recipe were not included in the repository. The source uses Indian rupees/lakhs and has a known filename mismatch: it loads `car_price_model.pkl`, while the preserved artifact is named `car_price_model (1).pkl`. That historical defect is intentionally unchanged.

`manifest.json` records the original commit, checkout byte hashes, and portable integrity hashes. Text integrity hashes normalize CRLF to LF only; binary hashes use the exact bytes. Tests verify preservation without deserializing the model or explainer.

Do not import this directory, install its requirements into production, or include its artifacts in production builds. No runnable or reproducible legacy environment is promised.
