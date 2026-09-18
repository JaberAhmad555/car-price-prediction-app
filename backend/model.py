"""Load only an operator-provisioned, checksum-verified Bangladesh artifact bundle."""

import hashlib
import json
import os
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "mendeley-fmb4xmp4k5-v2"
TARGET = "Estimated Bangladesh Market Listing Value"


class ModelStore:
    def __init__(self, directory=None):
        self.pipeline = None
        self.metadata = None
        self.error = "The reviewed Bangladesh model bundle is unavailable. Check deployment artifacts and runtime versions."
        directory = Path(directory or os.getenv("CARVALUE_MODEL_DIR") or ROOT / "artifacts/valuation").resolve()
        if not directory.is_relative_to((ROOT / "artifacts").resolve()):
            self.error = "Model directory must be inside the approved artifacts directory."
            return
        try:
            manifest_path = directory / "manifest.json"
            artifact = directory / "pipeline.joblib"
            if not manifest_path.resolve().is_relative_to(directory) or not artifact.resolve().is_relative_to(directory):
                raise ValueError("Artifact symlinks outside the bundle are prohibited")
            if not manifest_path.exists() or not artifact.exists():
                return
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (metadata.get("dataset_id") != SOURCE_ID or metadata.get("market") != "BD"
                    or metadata.get("currency") != "BDT" or metadata.get("target") != TARGET
                    or metadata.get("artifact_schema_version") != 1 or metadata.get("dataset_rows", 0) <= 0):
                raise ValueError("Incompatible dataset or model contract")
            if metadata["sklearn_version"] != sklearn.__version__:
                raise ValueError("Model and runtime scikit-learn versions differ")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["pipeline_sha256"]:
                raise ValueError("Artifact checksum does not match")
            self.pipeline = joblib.load(artifact)
            self.metadata = metadata
            self.error = None
        except (ValueError, KeyError, OSError, EOFError, ImportError):
            self.error = "Model bundle failed validation. Rebuild it with the documented training command."

    @property
    def ready(self):
        return self.pipeline is not None and self.metadata is not None
