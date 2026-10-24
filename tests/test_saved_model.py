"""Test saving and reloading the QuickBite classification pipeline."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, load_and_split


def test_saved_model_predictions(tmp_path):
    """Reloading a saved model should preserve its predictions."""
    X_train, X_test, y_train, _ = load_and_split()

    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)

    sample = X_test.iloc[:5]
    original_probabilities = pipeline.predict_proba(sample)[:, 1]

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)
    loaded_probabilities = loaded_pipeline.predict_proba(sample)[:, 1]

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )
