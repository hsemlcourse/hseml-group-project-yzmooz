import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid

from src.modeling import evaluate_model
from src.preprocessing import preprocess_features
from src.train_cp1 import build_preprocessor
from src.train_cp2 import cp2_search_spaces


def test_preprocess_features_clips_invalid_values_and_adds_features() -> None:
    df = pd.DataFrame(
        {
            "customer_age": [30],
            "product_price": [-10.0],
            "discount_percent": [-5.0],
            "product_rating": [9.0],
            "past_purchase_count": [4],
            "past_return_rate": [-0.2],
            "delivery_delay_days": [2],
            "session_length_minutes": [-1.0],
            "num_product_views": [-3],
            "device_type": ["mobile"],
            "product_category": ["electronics"],
            "shipping_method": ["express"],
            "payment_method": ["credit_card"],
            "used_coupon": [1],
        }
    )

    processed = preprocess_features(df)

    assert processed.loc[0, "product_price"] == 0.0
    assert processed.loc[0, "discount_percent"] == 0.0
    assert processed.loc[0, "product_rating"] == 5.0
    assert processed.loc[0, "past_return_rate"] == 0.0
    assert processed.loc[0, "session_length_minutes"] == 0.0
    assert processed.loc[0, "num_product_views"] == 0
    assert processed.loc[0, "invalid_product_price"] == 1
    assert processed.loc[0, "invalid_discount_percent"] == 1
    assert processed.loc[0, "invalid_product_rating"] == 1
    assert processed.loc[0, "invalid_past_return_rate"] == 1
    assert processed.loc[0, "invalid_session_length"] == 1
    assert processed.loc[0, "invalid_num_product_views"] == 1
    assert "price_after_discount" in processed.columns
    assert "discount_amount" in processed.columns
    assert "views_per_minute" in processed.columns
    assert "high_past_return_rate" in processed.columns
    assert "has_discount" in processed.columns


def test_build_preprocessor_detects_numeric_and_categorical_columns() -> None:
    x_sample = pd.DataFrame(
        {
            "customer_age": [25, 40],
            "product_price": [100.0, 250.0],
            "discount_percent": [10.0, 0.0],
            "product_rating": [4.2, 3.8],
            "past_purchase_count": [2, 8],
            "past_return_rate": [0.1, 0.4],
            "delivery_delay_days": [1, 5],
            "session_length_minutes": [12.0, 18.0],
            "num_product_views": [4, 6],
            "device_type": ["mobile", "desktop"],
            "product_category": ["electronics", "clothing"],
            "shipping_method": ["express", "standard"],
            "payment_method": ["credit_card", "paypal"],
            "used_coupon": [1, 0],
        }
    )

    preprocessor = build_preprocessor(
        x_sample,
        apply_feature_engineering=True,
    )

    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_evaluate_model_returns_expected_metric_keys() -> None:
    x_data = pd.DataFrame({"feature": [0, 1, 0, 1]})
    y_true = pd.Series([0, 1, 0, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(x_data, y_true)

    metrics = evaluate_model(model, x_data, y_true)

    assert set(metrics) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
    }


def test_cp2_search_spaces_cover_required_hyperparameter_grids() -> None:
    search_spaces = cp2_search_spaces()

    families = {space["family"] for space in search_spaces}
    total_runs = sum(len(ParameterGrid(space["grid"])) for space in search_spaces)

    assert families == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
    }
    assert total_runs == 38
