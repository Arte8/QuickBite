"""Tests for QuickBite cost calculations and threshold selection."""

import numpy as np

from latedeliveries.evaluation import calculate_cost, find_best_threshold


def test_no_flags_cost():
    """Missing two late orders costs $30."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    assert calculate_cost(y_true, y_pred) == 30


def test_all_flags_cost():
    """Flagging four orders costs $16."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1])

    assert calculate_cost(y_true, y_pred) == 16


def test_mixed_predictions_cost():
    """One flagged order and one missed late order costs $19."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert calculate_cost(y_true, y_pred) == 19


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
