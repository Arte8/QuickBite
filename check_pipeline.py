from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    build_pipeline,
    load_and_split,
)

# Load and split data
X_train, X_test, y_train, y_test = load_and_split()

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training late rate:", y_train.mean())

# Build and train pipeline using training data only
pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))

pipeline.fit(X_train, y_train)

# Create an order with missing preparation time
# and an unseen weather category.
new_order = X_train.iloc[[0]].copy()

new_order["prep_minutes"] = float("nan")
new_order["weather"] = "hail"

# Predict probability of a late delivery
probability = pipeline.predict_proba(new_order)[0, 1]

print("Predicted late probability:", probability)
