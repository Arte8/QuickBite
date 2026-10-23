"""Compare QuickBite late-delivery classification models."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from latedeliveries.pipeline import (
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
    "precision": "precision",
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


def main():
    """Compare models and tune Logistic Regression."""

    X_train, _X_test, y_train, _y_test = load_and_split()

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


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
