"""Tests for QuickBite cost calculations and threshold selection."""

import numpy as np
import pytest

from latedeliveries.evaluation import calculate_cost, find_best_threshold


@pytest.mark.parametrize(
    ("predictions", "expected_cost"),
    [
        ([0, 0, 0, 0], 30),  # Flag nobody
        ([1, 1, 1, 1], 16),  # Flag everybody
        ([1, 0, 0, 0], 19),  # One flag and one missed late order
    ],
)
def test_calculate_cost(predictions, expected_cost):
    """Check voucher costs for different prediction scenarios."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array(predictions)

    assert calculate_cost(y_true, y_pred) == expected_cost


def test_best_threshold():
    """Threshold selection should minimize voucher policy cost."""
    y_true = np.array([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.6, 0.3, 0.1])

    best_threshold, results = find_best_threshold(
        y_true,
        probabilities,
    )

    assert np.isclose(best_threshold, 0.15)
    assert min(row["cost"] for row in results) == 12


def test_threshold_prefers_flagging_nobody():
    """Flagging nobody should be cheapest when no orders are late."""
    y_true = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    best_threshold, results = find_best_threshold(
        y_true,
        probabilities,
    )

    predictions = (probabilities >= best_threshold).astype(int)

    assert predictions.sum() == 0
    assert calculate_cost(y_true, predictions) == 0


def test_threshold_prefers_flagging_everybody():
    """Flagging everybody should be cheapest when all orders are late."""
    y_true = np.array([1, 1, 1, 1])
    probabilities = np.array([0.6, 0.7, 0.8, 0.9])

    best_threshold, results = find_best_threshold(
        y_true,
        probabilities,
    )

    predictions = (probabilities >= best_threshold).astype(int)

    assert predictions.sum() == 4
    assert calculate_cost(y_true, predictions) == 16
