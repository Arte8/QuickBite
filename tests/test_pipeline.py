"""Tests for the QuickBite preprocessing and classification pipeline."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, load_and_split


def test_data_split():
    """Check that training and testing contain all 3000 orders."""
    X_train, X_test, y_train, y_test = load_and_split()

    assert len(X_train) == 2250
    assert len(X_test) == 750
    assert len(y_train) == 2250
    assert len(y_test) == 750


def test_excluded_features():
    """Ensure target leakage and identifier columns are excluded."""
    X_train, X_test, _, _ = load_and_split()

    for column in ["customer_rating", "order_id", "late"]:
        assert column not in X_train.columns
        assert column not in X_test.columns


def test_prediction_probabilities():
    """Verify the model produces valid probabilities."""
    X_train, X_test, y_train, _ = load_and_split()

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(X_test)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_missing_numeric_values():
    """Check that the pipeline handles missing numeric values."""
    X_train, X_test, y_train, _ = load_and_split()

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    sample = X_test.iloc[:1].copy()
    sample.loc[:, "distance_km"] = np.nan

    probabilities = pipeline.predict_proba(sample)[:, 1]

    assert np.isfinite(probabilities).all()


def test_unseen_category():
    """Check that an unseen category does not break predictions."""
    X_train, X_test, y_train, _ = load_and_split()

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    sample = X_test.iloc[:1].copy()
    sample.loc[:, "weather"] = "unknown_weather"

    probabilities = pipeline.predict_proba(sample)[:, 1]

    assert np.isfinite(probabilities).all()
