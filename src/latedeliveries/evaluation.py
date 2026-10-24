"""Cost calculation and threshold optimization."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Financial costs from the QuickBite case study
VOUCHER_COST = 4
MISSED_LATE_COST = 15

# Candidate thresholds: 0.05, 0.10, ..., 0.95
THRESHOLDS = np.arange(5, 100, 5) / 100


def calculate_cost(y_true, y_pred):
    """Calculate the total cost of delivery decisions."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Every flagged order costs $4
    flagged = np.sum(y_pred == 1)

    # Every late order not flagged costs $15
    missed_late = np.sum((y_true == 1) & (y_pred == 0))

    total_cost = VOUCHER_COST * flagged + MISSED_LATE_COST * missed_late

    return int(total_cost)


def find_best_threshold(y_true, probabilities):
    """Find the threshold with the lowest total cost."""

    results = []

    for threshold in THRESHOLDS:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)

        cost = calculate_cost(y_true, predictions)

        results.append(
            {
                "threshold": float(threshold),
                "cost": cost,
            }
        )

    # Select the threshold with the lowest cost.
    # Ties choose the smaller threshold.
    best = min(results, key=lambda row: row["cost"])

    return best["threshold"], results


def classification_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a chosen threshold."""

    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "total_cost": calculate_cost(y_true, predictions),
    }
