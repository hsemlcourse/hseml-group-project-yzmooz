from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    CP2_EXPERIMENTS_PATH,
    CP2_MODEL_PATH,
    RANDOM_STATE,
    RAW_TRAIN_PATH,
    ensure_raw_data_exists,
)
from src.modeling import evaluate_model
from src.train_cp1 import build_preprocessor, make_pipeline, split_data

RESULT_COLUMNS = [
    "model",
    "family",
    "params",
    "split",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
]


def cp2_search_spaces() -> list[dict]:
    """Return CP2 model families and hyperparameter grids."""
    return [
        {
            "family": "logistic_regression",
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "grid": {
                "C": [0.3, 1.0, 3.0],
                "class_weight": [None, "balanced"],
            },
        },
        {
            "family": "decision_tree",
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {
                "max_depth": [4, 8, 12, None],
                "min_samples_leaf": [50, 200],
            },
        },
        {
            "family": "random_forest",
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [20, 100],
            },
        },
        {
            "family": "extra_trees",
            "estimator": ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [20, 100],
            },
        },
    ]


def format_model_name(family: str, run_number: int) -> str:
    return f"{family}_grid_{run_number:02d}"


def build_tuned_model(
    *,
    estimator,
    params: dict,
    engineered_preprocessor,
):
    estimator = clone(estimator).set_params(**params)
    return make_pipeline(
        preprocessor=clone(engineered_preprocessor),
        estimator=estimator,
        apply_feature_engineering=True,
    )


def add_result_row(
    results: list[dict],
    *,
    metrics: dict[str, float],
    model_name: str,
    family: str,
    params: dict,
    split: str,
) -> None:
    row = dict(metrics)
    row["model"] = model_name
    row["family"] = family
    row["params"] = repr(params)
    row["split"] = split
    results.append(row)


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

    results = []

    baseline = make_pipeline(
        preprocessor=clone(raw_preprocessor),
        estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        apply_feature_engineering=False,
    )
    print("Training baseline reference: logistic_regression_baseline...")
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate_model(baseline, x_val, y_val)
    add_result_row(
        results,
        metrics=baseline_metrics,
        model_name="logistic_regression_baseline",
        family="baseline",
        params={"feature_engineering": False, "class_weight": "balanced"},
        split="validation",
    )

    best_model = baseline
    best_model_name = "logistic_regression_baseline"
    best_family = "baseline"
    best_params = {"feature_engineering": False, "class_weight": "balanced"}
    best_score = baseline_metrics["roc_auc"]

    for search_space in cp2_search_spaces():
        family = search_space["family"]
        estimator = search_space["estimator"]

        for run_number, params in enumerate(ParameterGrid(search_space["grid"]), 1):
            model_name = format_model_name(family, run_number)
            print(f"Training {model_name}: {params}")
            model = build_tuned_model(
                estimator=estimator,
                params=params,
                engineered_preprocessor=engineered_preprocessor,
            )
            model.fit(x_train, y_train)

            val_metrics = evaluate_model(model, x_val, y_val)
            add_result_row(
                results,
                metrics=val_metrics,
                model_name=model_name,
                family=family,
                params=params,
                split="validation",
            )

            if val_metrics["roc_auc"] > best_score:
                best_score = val_metrics["roc_auc"]
                best_model = model
                best_model_name = model_name
                best_family = family
                best_params = params

    results_df = pd.DataFrame(results)
    validation_results_df = results_df.sort_values("roc_auc", ascending=False)

    print("\nTop validation results:")
    print(validation_results_df.head(10)[RESULT_COLUMNS])

    test_metrics = evaluate_model(best_model, x_test, y_test)
    add_result_row(
        results,
        metrics=test_metrics,
        model_name=best_model_name,
        family=best_family,
        params=best_params,
        split="test",
    )

    final_results_df = pd.DataFrame(results)[RESULT_COLUMNS]
    final_results_df = final_results_df.sort_values(
        ["split", "roc_auc"],
        ascending=[False, False],
    )

    Path("report").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    final_results_df.to_csv(CP2_EXPERIMENTS_PATH, index=False)
    joblib.dump(best_model, CP2_MODEL_PATH)

    print("\nBest CP2 model:", best_model_name)
    print("Best family:", best_family)
    print("Best params:", best_params)
    print("\nTest metrics:")
    print(pd.DataFrame([test_metrics]))
    print(f"\nSaved experiments to {CP2_EXPERIMENTS_PATH}")
    print(f"Saved model to {CP2_MODEL_PATH}")


if __name__ == "__main__":
    main()
