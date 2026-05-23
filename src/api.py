from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CP2_MODEL_PATH

THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / CP2_MODEL_PATH

EXPECTED_FEATURES = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views",
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method",
    "used_coupon",
]

CATEGORY_VALUES = {
    "device_type": ["desktop", "mobile", "tablet"],
    "product_category": ["beauty", "clothing", "electronics", "home", "sports", "toys"],
    "shipping_method": ["express", "same_day", "standard"],
    "payment_method": ["apple_pay", "credit_card", "debit_card", "paypal"],
}


class PredictionRequest(BaseModel):
    customer_age: int = Field(..., examples=[25])
    product_price: float = Field(..., examples=[11.67])
    discount_percent: float = Field(..., examples=[43.56])
    product_rating: float = Field(..., examples=[1.93])
    past_purchase_count: int = Field(..., examples=[8])
    past_return_rate: float = Field(..., examples=[0.33])
    delivery_delay_days: float = Field(..., examples=[0.0])
    session_length_minutes: float = Field(..., examples=[3.27])
    num_product_views: int = Field(..., examples=[26])
    device_type: Literal["desktop", "mobile", "tablet"] = Field(
        ...,
        examples=["tablet"],
    )
    product_category: Literal[
        "beauty",
        "clothing",
        "electronics",
        "home",
        "sports",
        "toys",
    ] = Field(..., examples=["sports"])
    shipping_method: Literal["express", "same_day", "standard"] = Field(
        ...,
        examples=["express"],
    )
    payment_method: Literal["apple_pay", "credit_card", "debit_card", "paypal"] = Field(
        ...,
        examples=["paypal"],
    )
    used_coupon: Literal[0, 1] = Field(..., examples=[1])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{feature: getattr(self, feature) for feature in EXPECTED_FEATURES}]
        )


class PredictionResponse(BaseModel):
    prediction: int
    returned_probability: float
    threshold: float


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run python src/train_cp2.py "
            "or add models/cp2_best_model.joblib before starting the API."
        )
    return joblib.load(MODEL_PATH)


def is_model_available() -> bool:
    return MODEL_PATH.exists()


app = FastAPI(
    title="Product Return Prediction API",
    description="Predicts whether an e-commerce order is likely to be returned.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": is_model_available()}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_path": str(MODEL_PATH),
        "model_loaded": is_model_available(),
        "model_name": "random_forest_grid_08",
        "threshold": THRESHOLD,
        "expected_features": EXPECTED_FEATURES,
        "category_values": CATEGORY_VALUES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = payload.to_frame()
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        returned_probability=probability,
        threshold=THRESHOLD,
    )
