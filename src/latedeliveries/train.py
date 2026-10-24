"""Compare QuickBite late-delivery classification models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.evaluation import (
    calculate_cost,
    classification_metrics,
    find_best_threshold,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    load_and_split,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Five-fold stratified cross-validation
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Models to compare
MODELS = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "BalancedLogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    ),
}

# Evaluation metrics


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compare_models(X_train, y_train):
    """Compare models using training-only cross-validation."""

    results = []

    for name, model in MODELS.items():
        print(f"Evaluating {name}...")

        # All models use the same preprocessing pipeline.
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
        )

        row = {"model": name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        results.append(row)

    return pd.DataFrame(results)


def tune_model(X_train, y_train):
    """Tune Logistic Regression using training data only."""

    pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest hyperparameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation ROC-AUC:")
    print(grid_search.best_score_)

    return grid_search


def optimize_threshold(X_train, y_train, grid_search):
    """Find the lowest-cost threshold using training data."""

    best_pipeline = grid_search.best_estimator_

    # Obtain out-of-fold probabilities.
    probabilities = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=CV,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Calculate costs across the threshold grid.
    best_threshold, threshold_results = find_best_threshold(
        y_train,
        probabilities,
    )

    best_cost = min(result["cost"] for result in threshold_results)

    print("\nBest cost-based threshold:", best_threshold)
    print("Minimum out-of-fold training cost: $", best_cost)

    return best_threshold, threshold_results


def evaluate_final_model(X_train, X_test, y_train, y_test, grid_search, best_threshold):
    """Fit final pipeline and evaluate once on the test set."""

    # Fit the selected pipeline on all training data.
    final_pipeline = grid_search.best_estimator_
    final_pipeline.fit(X_train, y_train)

    # Evaluate the untouched test set.
    probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(
        y_test,
        probabilities,
        best_threshold,
    )

    # Compare three voucher policies.
    nobody_cost = calculate_cost(
        y_test,
        np.zeros(len(y_test), dtype=int),
    )

    everybody_cost = calculate_cost(
        y_test,
        np.ones(len(y_test), dtype=int),
    )

    model_cost = metrics["total_cost"]

    metrics["policy_costs"] = {
        "flag_nobody": nobody_cost,
        "flag_everybody": everybody_cost,
        "model": model_cost,
    }

    print("\nFinal test-set results:")
    print(json.dumps(metrics, indent=4))

    # Save evaluation results.
    with (REPORTS_DIR / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=4)

    return final_pipeline, metrics


def save_model(final_pipeline, best_threshold, grid_search, test_metrics):
    """Save the fitted pipeline and model metadata."""

    # Create models directory
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the complete trained pipeline
    model_path = models_dir / "late_orders.joblib"
    joblib.dump(final_pipeline, model_path)

    # Save model information
    model_info = {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "threshold": float(best_threshold),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_parameters": grid_search.best_params_,
        "test_metrics": test_metrics,
    }

    info_path = models_dir / "model_info.json"

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved metadata:", info_path)


def main():
    """Compare models and tune Logistic Regression."""
    X_train, X_test, y_train, y_test = load_and_split()
    # Compare models
    results = compare_models(X_train, y_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORTS_DIR / "cv_results.csv",
        index=False,
    )

    print("\nCross-validation results:")
    print(results.round(3).to_string(index=False))

    # Tune selected model
    grid_search = tune_model(X_train, y_train)
    # Select the cost-minimizing threshold.
    best_threshold, threshold_results = optimize_threshold(
        X_train,
        y_train,
        grid_search,
    )

    # Compare tuned and untuned ROC-AUC
    untuned = results.loc[results["model"] == "LogisticRegression"].iloc[0]

    improvement = grid_search.best_score_ - untuned["roc_auc_mean"]

    print("\nUntuned ROC-AUC:", untuned["roc_auc_mean"])
    print("Tuned ROC-AUC:", grid_search.best_score_)
    print("Improvement:", improvement)
    print("Fold-to-fold standard deviation:", untuned["roc_auc_std"])

    if improvement > untuned["roc_auc_std"]:
        print("Improvement exceeds the fold-to-fold spread.")
    else:
        print("Improvement does not exceed the fold-to-fold spread.")

    # Evaluate the final model on the untouched test set.
    final_pipeline, test_metrics = evaluate_final_model(
        X_train,
        X_test,
        y_train,
        y_test,
        grid_search,
        best_threshold,
    )

    # Save the trained model and metadata
    save_model(
        final_pipeline,
        best_threshold,
        grid_search,
        test_metrics,
    )


if __name__ == "__main__":
    main()
