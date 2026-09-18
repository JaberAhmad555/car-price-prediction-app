"""Public valuation and ephemeral photo-quality API for CarValue BD."""

from contextlib import asynccontextmanager
import io
import math
import os
import warnings

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .model import ModelStore, TARGET

MAX_IMAGE_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    brand: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=1950, le=2100, strict=True)
    mileage_km: int = Field(ge=0, le=1_000_000, strict=True)
    transmission: str = Field(min_length=1, max_length=40)
    fuel_type: str = Field(min_length=1, max_length=40)
    body_type: str = Field(min_length=1, max_length=40)
    engine_cc: int = Field(ge=400, le=10000, strict=True)

    @field_validator("brand", "model", "transmission", "fuel_type", "body_type")
    @classmethod
    def normalize_category(cls, value):
        return " ".join(value.casefold().split())


class PredictionResponse(BaseModel):
    estimated_price_bdt: int
    estimated_price_lakh: float
    lower_estimate: int
    upper_estimate: int
    range_method: str
    model_version: str
    dataset_description: str
    target: str = TARGET
    historical: bool = True
    vehicle: PredictionRequest
    disclaimer: str


@asynccontextmanager
async def lifespan(app):
    app.state.model = ModelStore()
    if os.getenv("REQUIRE_MODEL") == "1" and not app.state.model.ready:
        raise RuntimeError(app.state.model.error)
    yield


app = FastAPI(title="CarValue BD API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=[origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def store(request):
    return request.app.state.model


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "service": "CarValue BD", "model_ready": store(request).ready}


@app.get("/api/metadata")
def metadata(request: Request):
    model = store(request)
    public = {key: model.metadata[key] for key in (
        "selected_model", "model_version", "dataset_rows", "raw_rows", "test_metrics", "validation_metrics",
        "range_method", "split_rows", "split_method", "trained_on", "dataset_description",
    )} if model.ready else {}
    return {"name": "CarValue BD", "target": TARGET, "currency": "BDT", "historical": True,
            "model_ready": model.ready, "status_message": model.error,
            "dataset_source": "https://data.mendeley.com/datasets/fmb4xmp4k5/2",
            "dataset_doi": "10.17632/fmb4xmp4k5.2", "license": "CC BY 4.0",
            "publication_date": "2024-01-02", "publisher_reported_rows": 1209,
            "dataset_rows": None, "test_metrics": None, "selected_model": None,
            "photo_analysis": "Image quality only; no damage recognition or monetary adjustment", **public}


@app.get("/api/options")
def options(request: Request):
    model = store(request)
    return {"available": model.ready, **(model.metadata["options"] if model.ready else {
        "brand": [], "model": [], "models_by_brand": {}, "transmission": [], "fuel_type": [], "body_type": [], "bounds": {},
    })}


@app.get("/api/insights")
def insights(request: Request):
    model = store(request)
    return {"available": model.ready, "historical": True,
            "data": model.metadata["insights"] if model.ready else None,
            "description": "Calculated from usable historical listing rows. Descriptive associations, not sale prices or causal effects."}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = store(request)
    if not model.ready:
        raise HTTPException(503, detail=model.error)
    data = payload.model_dump()
    options = model.metadata["options"]
    for field in ("brand", "transmission", "fuel_type", "body_type"):
        if data[field] not in options[field]:
            raise HTTPException(422, detail=f"Unsupported {field.replace('_', ' ')} in the verified dataset.")
    if data["model"] not in options["models_by_brand"].get(data["brand"], []):
        raise HTTPException(422, detail="This brand/model pair is outside the verified dataset.")
    for field, bounds in options["bounds"].items():
        if not bounds["min"] <= data[field] <= bounds["max"]:
            raise HTTPException(422, detail=f"{field.replace('_', ' ')} is outside historical coverage ({bounds['min']}–{bounds['max']}).")
    value = float(model.pipeline.predict(pd.DataFrame([data]))[0])
    if not math.isfinite(value) or value <= 0:
        raise HTTPException(503, detail="The model could not produce a valid estimate for this vehicle.")
    low, high = model.metadata["residual_offsets_bdt"]
    return PredictionResponse(
        estimated_price_bdt=round(value), estimated_price_lakh=round(value / 100000, 2),
        lower_estimate=max(0, round(min(value, value + low))), upper_estimate=round(max(value, value + high)),
        range_method=model.metadata["range_method"] + " Bounds include the point estimate and are clipped at zero.",
        model_version=model.metadata["model_version"], dataset_description=model.metadata["dataset_description"],
        vehicle=payload, disclaimer="This AI estimate represents an expected market listing value based on historical Bangladesh used-car data. It is not a guaranteed sale or transaction price.")


def inspect_bytes(content):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "Use a JPEG, PNG or WebP image.")
                if original.width * original.height > 16_000_000:
                    raise HTTPException(413, "Image exceeds 16 megapixels. Resize it before uploading.")
                image = ImageOps.exif_transpose(original).convert("RGB")
                width, height = image.size
                image.thumbnail((1280, 1280))
                gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(422, "The file could not be safely decoded as an image.") from None
    checks = [
        {"name": "Resolution", "passed": min(width, height) >= 480 and max(width, height) >= 640,
         "detail": f"{width} × {height} px. Aim for at least 640 × 480."},
        {"name": "Lighting", "passed": 40 <= brightness <= 220,
         "detail": "Balanced exposure" if 40 <= brightness <= 220 else "Try more even daylight; this photo may be too dark or overexposed."},
        {"name": "Sharpness", "passed": sharpness >= 90,
         "detail": "Sufficient edge detail" if sharpness >= 90 else "Potential blur or low texture. Hold the camera steady and try again."},
    ]
    return {"width": width, "height": height, "checks": checks,
            "brightness_mean": round(brightness, 2), "laplacian_variance": round(sharpness, 2),
            "quality_only": True, "damage_detection_available": False,
            "notice": "Heuristic photo-quality checks only. Vehicle presence, damage and mechanical condition are not assessed. No price adjustment is applied.",
            "retention": "Processed in memory; images are not stored or used for training."}


@app.post("/api/inspect")
async def inspect(request: Request):
    if request.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Send one JPEG, PNG or WebP image as the request body.")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Images must be smaller than 8 MB.")
    if not content:
        raise HTTPException(422, "The image is empty.")
    return await run_in_threadpool(inspect_bytes, bytes(content))
