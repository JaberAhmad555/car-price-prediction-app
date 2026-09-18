"""Static regression guards; never import or deserialize historical artifacts."""

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "legacy/manifest.json").read_text(encoding="utf-8"))
ARTIFACT_HASHES = {
    item["sha256"] for item in MANIFEST["files"] if item["hash_mode"] == "bytes"
}
LEGACY_REFERENCE = re.compile(
    r"(?i)(?<![a-z0-9_])legacy(?![a-z0-9_])"
    r"|car_price_model(?:\s*\(1\))?\.pkl|shap_explainer\.pkl"
)
CODE_SUFFIXES = {
    ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".json", ".toml", ".yaml", ".yml", ".sh", ".bash", ".ps1",
    ".bat", ".cmd", ".ini", ".cfg", ".conf",
}
IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".next", ".turbo",
}


def legacy_violations(root, artifact_hashes=ARTIFACT_HASHES):
    """Reject literal runtime/build dependencies and renamed artifact copies.

    Documentation, data values, and tests may discuss legacy. Their text is not
    treated as executable, but artifacts copied into docs/data are still caught.
    This deliberately conservative scan cannot prove safety of arbitrary dynamic
    code; future runtime loaders also need trusted model manifests.
    """
    violations = []
    for directory, child_dirs, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        child_dirs[:] = sorted(name for name in child_dirs if name not in IGNORED_DIRS)
        if parent == root:
            child_dirs[:] = [name for name in child_dirs if name not in {"legacy", "tests"}]
        for name in list(child_dirs):
            path = parent / name
            if path.is_symlink():
                violations.append(str(path.relative_to(root)) + ": directory symlink needs review")
                child_dirs.remove(name)
        for name in sorted(filenames):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                violations.append(str(relative) + ": file symlink needs review")
                continue
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() in artifact_hashes:
                violations.append(str(relative) + ": copied historical artifact")
            if relative.parts[0] in {"docs", "data"}:
                continue
            is_runtime_file = (
                path.suffix.lower() in CODE_SUFFIXES
                or name.lower().startswith(("requirements", "dockerfile"))
                or name in {"Makefile", "Procfile"}
            )
            if is_runtime_file and LEGACY_REFERENCE.search(content.decode("utf-8", errors="replace")):
                violations.append(str(relative) + ": legacy runtime/build reference")
    return violations


class LegacyIsolationTests(unittest.TestCase):
    def test_production_and_build_paths_have_no_legacy_dependencies(self):
        self.assertEqual(legacy_violations(ROOT), [])

    def test_original_files_remain_recoverable_with_matching_hashes(self):
        self.assertEqual(
            {item["path"] for item in MANIFEST["files"]},
            {"app.py", "requirements.txt", "car_price_model (1).pkl", "shap_explainer.pkl"},
        )
        for item in MANIFEST["files"]:
            with self.subTest(file=item["path"]):
                content = (ROOT / "legacy" / item["path"]).read_bytes()
                if item["hash_mode"] == "normalize_crlf_to_lf":
                    content = content.replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_legacy_artifacts_are_absent_from_repository_root(self):
        for name in ("car_price_model.pkl", "car_price_model (1).pkl", "shap_explainer.pkl"):
            with self.subTest(file=name):
                self.assertFalse((ROOT / name).exists())

    def test_guard_rejects_imports_loaders_frontend_and_build_references(self):
        cases = [
            ("backend/main.py", "from legacy.app import model"),
            ("backend/main.py", "import legacy.app"),
            ("backend/main.py", "importlib.import_module('legacy.app')"),
            ("ml/valuation/load.py", "joblib.load('../legacy/car_price_model (1).pkl')"),
            ("backend/main.py", "pickle.load(open('shap_explainer.pkl', 'rb'))"),
            ("frontend/src/load.ts", "const path = '../legacy/car_price_model (1).pkl';"),
            ("requirements.txt", "-r legacy/requirements.txt"),
            (".github/workflows/release.yml", "run: cp legacy/app.py backend/main.py"),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content):
                with tempfile.TemporaryDirectory(prefix="bd-contract-test-") as directory:
                    root = Path(directory)
                    path = root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    self.assertTrue(legacy_violations(root))

    def test_guard_detects_renamed_artifact_bytes(self):
        payload = b"synthetic historical artifact bytes; not a pickle"
        hashes = {hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory(prefix="bd-contract-test-") as directory:
            root = Path(directory)
            (root / "backend").mkdir()
            (root / "backend/renamed.bin").write_bytes(payload)
            self.assertTrue(legacy_violations(root, hashes))

    def test_guard_allows_independent_code_and_historical_documentation(self):
        with tempfile.TemporaryDirectory(prefix="bd-contract-test-") as directory:
            root = Path(directory)
            (root / "backend").mkdir()
            (root / "backend/main.py").write_text("from valuation import predictor\n", encoding="utf-8")
            (root / "README.md").write_text("Historical material is in legacy/.", encoding="utf-8")
            self.assertEqual(legacy_violations(root), [])


if __name__ == "__main__":
    unittest.main()
