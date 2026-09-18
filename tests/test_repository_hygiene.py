"""Minimal checks for empty secret placeholders and Git ignore boundaries."""

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def test_environment_template_contains_only_empty_values(self):
        entries = []
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            self.assertEqual(separator, "=")
            self.assertTrue(key)
            self.assertEqual(value.strip(), "", msg="Environment template must not contain values")
            entries.append(key)
        self.assertTrue(entries)
        self.assertEqual(len(entries), len(set(entries)))

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "Requires a Git checkout")
    def test_secrets_data_and_artifacts_are_ignored_but_contracts_are_trackable(self):
        ignored = [
            ".env", ".env.production", "backend/.env.local", ".venv/example.py",
            "data/raw/example.json", "data/processed/example.parquet",
            "data/fixtures/accidental.csv", "ml/valuation/generated.joblib",
            "uploads/photo.jpg", "backend/__pycache__/main.pyc",
        ]
        trackable = [
            ".env.example", "data/schemas/market_record.schema.json",
            "data/manifests/.gitkeep", "data/fixtures/.gitkeep",
            "legacy/car_price_model (1).pkl", "legacy/shap_explainer.pkl",
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            input=("\0".join(ignored + trackable) + "\0").encode("utf-8"),
            capture_output=True, cwd=ROOT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}
        self.assertEqual(actual, set(ignored))


if __name__ == "__main__":
    unittest.main()
