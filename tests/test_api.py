from fastapi.testclient import TestClient

from src.api import app

SAMPLE_PAYLOAD = {
    "customer_age": 25,
    "product_price": 11.67,
    "discount_percent": 43.56,
    "product_rating": 1.93,
    "past_purchase_count": 8,
    "past_return_rate": 0.33,
    "delivery_delay_days": 0.0,
    "session_length_minutes": 3.27,
    "num_product_views": 26,
    "device_type": "tablet",
    "product_category": "sports",
    "shipping_method": "express",
    "payment_method": "paypal",
    "used_coupon": 1,
}


def test_health_reports_model_available() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_contains_expected_features() -> None:
    client = TestClient(app)

    response = client.get("/model-info")
    data = response.json()

    assert response.status_code == 200
    assert data["model_loaded"] is True
    assert data["model_name"] == "random_forest_grid_08"
    assert len(data["expected_features"]) == 14


def test_predict_returns_probability_and_binary_prediction() -> None:
    client = TestClient(app)

    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()

    assert response.status_code == 200
    assert data["prediction"] in {0, 1}
    assert 0.0 <= data["returned_probability"] <= 1.0
    assert data["threshold"] == 0.5
