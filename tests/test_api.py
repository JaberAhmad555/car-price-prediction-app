"""Offline API integration checks. Test prices are never served by the app."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from backend.main import app
from backend.model import ModelStore, TARGET


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.payload = {"brand": "example", "model": "sample", "year": 2018, "mileage_km": 65000,
                        "transmission": "automatic", "fuel_type": "petrol", "body_type": "sedan", "engine_cc": 1500}

    def test_health_and_metadata_are_honest(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        metadata = self.client.get("/api/metadata").json()
        self.assertEqual(metadata["target"], TARGET)
        if not metadata["model_ready"]:
            self.assertIsNone(metadata["test_metrics"])
            self.assertIsNone(metadata["dataset_rows"])
            self.assertEqual(self.client.get("/api/options").json()["brand"], [])
            self.assertIsNone(self.client.get("/api/insights").json()["data"])

    def test_missing_model_returns_no_price(self):
        with patch.object(app.state, "model", SimpleNamespace(ready=False, error="Model pending")):
            response = self.client.post("/api/predict", json=self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("estimated_price_bdt", response.json())

    def test_provisioned_model_prediction_matches_saved_pipeline(self):
        model = app.state.model
        if not model.ready:
            self.skipTest("No locally provisioned Bangladesh bundle")
        options = self.client.get("/api/options").json()
        self.assertTrue(options["available"])
        brand = options["brand"][0]
        payload = {"brand": brand, "model": options["models_by_brand"][brand][0],
                   **{field: options[field][0] for field in ("transmission", "fuel_type", "body_type")},
                   **{field: (bounds["min"] + bounds["max"]) // 2 for field, bounds in options["bounds"].items()}}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        quote = response.json()
        expected = float(model.pipeline.predict(pd.DataFrame([payload]))[0])
        self.assertEqual(quote["estimated_price_bdt"], round(expected))
        self.assertEqual(quote["estimated_price_lakh"], round(expected / 100000, 2))
        self.assertLessEqual(quote["lower_estimate"], quote["estimated_price_bdt"])
        self.assertGreaterEqual(quote["upper_estimate"], quote["estimated_price_bdt"])
        self.assertEqual(quote["model_version"], model.metadata["model_version"])
        self.assertEqual(self.client.post("/api/predict", json={**payload, "model": "unsupported-model"}).status_code, 422)

    def test_invalid_requests_are_rejected(self):
        for updates in ({"mileage_km": -1}, {"year": 2018.5}, {"engine_cc": 0}, {"present_price": 42}, {"currency": "INR"}):
            with self.subTest(updates=updates):
                self.assertEqual(self.client.post("/api/predict", json={**self.payload, **updates}).status_code, 422)

    def test_prediction_contract_with_isolated_test_double(self):
        metadata = {"options": {"brand": ["example"], "transmission": ["automatic"], "fuel_type": ["petrol"], "body_type": ["sedan"],
            "models_by_brand": {"example": ["sample"]}, "bounds": {"year": {"min": 2000, "max": 2020}}},
            "residual_offsets_bdt": [-125000, 210000], "range_method": "Synthetic test residuals only",
            "model_version": "test-only", "dataset_description": "Invented isolated test double"}
        fake = SimpleNamespace(ready=True, metadata=metadata, pipeline=SimpleNamespace(predict=lambda frame: [2250000.]))
        with patch.object(app.state, "model", fake):
            response = self.client.post("/api/predict", json=self.payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["estimated_price_bdt"], 2250000)
            self.assertEqual(body["estimated_price_lakh"], 22.5)
            self.assertEqual(body["lower_estimate"], 2125000)
            self.assertEqual(body["upper_estimate"], 2460000)
            self.assertTrue(body["historical"])
            for updates in ({"model": "unsupported"}, {"year": 2025}, {"fuel_type": "unsupported"}):
                self.assertEqual(self.client.post("/api/predict", json={**self.payload, **updates}).status_code, 422)

    def test_image_quality_does_not_claim_damage_detection(self):
        stream = io.BytesIO()
        Image.new("RGB", (800, 600), (100, 100, 100)).save(stream, format="PNG")
        response = self.client.post("/api/inspect", content=stream.getvalue(), headers={"Content-Type": "image/png"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["quality_only"])
        self.assertFalse(result["damage_detection_available"])
        checks = {check["name"]: check["passed"] for check in result["checks"]}
        self.assertTrue(checks["Resolution"])
        self.assertTrue(checks["Lighting"])
        self.assertFalse(checks["Sharpness"])
        self.assertNotIn("condition_score", result)
        self.assertNotIn("price_adjustment", result)

    def test_invalid_and_oversized_images(self):
        self.assertEqual(self.client.post("/api/inspect", content=b"bad", headers={"Content-Type": "image/png"}).status_code, 422)
        self.assertEqual(self.client.post("/api/inspect", content=b"bad", headers={"Content-Type": "text/html"}).status_code, 415)
        self.assertEqual(self.client.post("/api/inspect", content=b"x" * (8 * 1024 * 1024 + 1), headers={"Content-Type": "image/png"}).status_code, 413)

    def test_cors_allows_local_frontend(self):
        response = self.client.options("/api/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")

    def test_unapproved_artifact_is_never_loaded(self):
        with patch("backend.model.joblib.load") as load:
            self.assertFalse(ModelStore(Path(__file__).parent.parent / "legacy").ready)
            load.assert_not_called()
