"""Compare QuickBite late-delivery classification models."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

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


def main():
    """Run the initial model comparison."""

    # Generate data and create the train/test split.
    X_train, _X_test, y_train, _y_test = load_and_split()

    # Compare models using training data only.
    results = compare_models(X_train, y_train)

    # Create reports directory if needed.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save results.
    output_file = REPORTS_DIR / "cv_results.csv"
    results.to_csv(output_file, index=False)

    print("\nCross-validation results:")
    print(results.round(3).to_string(index=False))

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
