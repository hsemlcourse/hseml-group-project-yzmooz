from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    CP1_EXPERIMENTS_PATH,
    CP1_MODEL_PATH,
    ID_COL,
    RANDOM_STATE,
    RAW_TRAIN_PATH,
    TARGET_COL,
    ensure_raw_data_exists,
)
from src.modeling import evaluate_model
from src.preprocessing import preprocess_features


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a reproducible stratified train/validation/test split."""
    x = data.drop(columns=[TARGET_COL, ID_COL])
    y = data[TARGET_COL]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_preprocessor(
    x_sample: pd.DataFrame,
    *,
    apply_feature_engineering: bool,
) -> ColumnTransformer:
    """Build a preprocessor for either raw or engineered features."""
    if apply_feature_engineering:
        x_sample = preprocess_features(x_sample)

    numeric_features = x_sample.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_sample.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def make_pipeline(
    *,
    preprocessor: ColumnTransformer,
    estimator,
    apply_feature_engineering: bool,
) -> Pipeline:
    """Assemble a modeling pipeline with optional feature engineering."""
    steps = []

    if apply_feature_engineering:
        steps.append(
            (
                "features",
                FunctionTransformer(preprocess_features, validate=False),
            )
        )

    steps.extend(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    return Pipeline(steps=steps)


def build_models(
    *,
    raw_preprocessor: ColumnTransformer,
    engineered_preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    """Create CP1 experiment pipelines."""
    return {
        "dummy_most_frequent": make_pipeline(
            preprocessor=raw_preprocessor,
            estimator=DummyClassifier(strategy="most_frequent"),
            apply_feature_engineering=False,
        ),
        "logistic_regression_baseline": make_pipeline(
            preprocessor=raw_preprocessor,
            estimator=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            apply_feature_engineering=False,
        ),
        "logistic_regression_features": make_pipeline(
            preprocessor=engineered_preprocessor,
            estimator=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            apply_feature_engineering=True,
        ),
        "decision_tree_depth6": make_pipeline(
            preprocessor=engineered_preprocessor,
            estimator=DecisionTreeClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            apply_feature_engineering=True,
        ),
        "random_forest_depth12": make_pipeline(
            preprocessor=engineered_preprocessor,
            estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            apply_feature_engineering=True,
        ),
    }


def main() -> None:
    ensure_raw_data_exists()
    data = pd.read_csv(RAW_TRAIN_PATH).drop_duplicates()
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

    raw_preprocessor = build_preprocessor(
        x_train.head(1000),
        apply_feature_engineering=False,
    )
    engineered_preprocessor = build_preprocessor(
        x_train.head(1000),
        apply_feature_engineering=True,
    )
    models = build_models(
        raw_preprocessor=raw_preprocessor,
        engineered_preprocessor=engineered_preprocessor,
    )

    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)

        val_metrics = evaluate_model(model, x_val, y_val)
        val_metrics["model"] = model_name
        val_metrics["split"] = "validation"
        results.append(val_metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["model", "split", "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    ].sort_values("roc_auc", ascending=False)

    print("\nValidation results:")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]

    test_metrics = evaluate_model(best_model, x_test, y_test)
    test_metrics["model"] = best_model_name
    test_metrics["split"] = "test"

    test_results_df = pd.DataFrame([test_metrics])
    final_results_df = pd.concat([results_df, test_results_df], ignore_index=True)

    Path("report").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    final_results_df.to_csv(CP1_EXPERIMENTS_PATH, index=False)
    joblib.dump(best_model, CP1_MODEL_PATH)

    print("\nBest model:", best_model_name)
    print("\nTest metrics:")
    print(test_results_df)


if __name__ == "__main__":
    main()
