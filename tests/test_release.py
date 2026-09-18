"""The committed inference bundle must work without local data or training code."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_portable_bundle_without_raw_data_or_training_modules(self):
        with tempfile.TemporaryDirectory(prefix="carvalue-release-") as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "backend", target / "backend", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "artifacts/valuation", target / "artifacts/valuation")
            env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "CARVALUE_MODEL_DIR"}}
            env["REQUIRE_MODEL"] = "1"
            result = subprocess.run([sys.executable, "-m", "backend.verify_release"], cwd=target,
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Release inference verified", result.stdout)
