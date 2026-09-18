"""Verify the shipped inference bundle and API without raw data or training."""

from fastapi.testclient import TestClient
from .main import app


def verify():
    with TestClient(app) as client:
        assert client.get("/health").json()["model_ready"], "Release model must load"
        metadata = client.get("/api/metadata").json()
        assert metadata["currency"] == "BDT" and metadata["historical"]
        options = client.get("/api/options").json()
        assert "axio" in options["models_by_brand"]["toyota"]
        # Specifications from an actual source record; no expected price is hardcoded.
        vehicle = {"brand": "toyota", "model": "axio", "year": 2016, "mileage_km": 96000,
                   "transmission": "automatic", "fuel_type": "cng, hybrid", "body_type": "saloon", "engine_cc": 1500}
        response = client.post("/api/predict", json=vehicle)
        assert response.status_code == 200, response.text
        quote = response.json()
        assert isinstance(quote["estimated_price_bdt"], int) and quote["estimated_price_bdt"] > 0
        assert quote["lower_estimate"] <= quote["estimated_price_bdt"] <= quote["upper_estimate"]
        print("Release inference verified:", metadata["model_version"], quote["estimated_price_bdt"], "BDT")


if __name__ == "__main__":
    verify()
