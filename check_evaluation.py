from latedeliveries.evaluation import (
    calculate_cost,
    find_best_threshold,
)

# Four example orders
y_true = [1, 0, 1, 0]

# Predicted probabilities of arriving late
probabilities = [0.80, 0.60, 0.30, 0.10]

# Test cost at threshold 0.50
predictions = [1, 1, 0, 0]

cost = calculate_cost(y_true, predictions)

print("Cost at threshold 0.50:", cost)

# Find lowest-cost threshold
best_threshold, results = find_best_threshold(y_true, probabilities)

print("Best threshold:", best_threshold)
print("Lowest cost:", min(row["cost"] for row in results))
