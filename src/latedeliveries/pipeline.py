"""Feature selection, data splitting, and preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

# Fixed random state for reproducibility
RANDOM_STATE = 42

# Numerical features available when the order is placed
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
]

# Complete feature list
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_split():
    """Generate orders and create a stratified 75/25 split."""

    df = make_orders()

    X = df[FEATURES]
    y = df["late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


def build_pipeline(model):
    """Build a preprocessing and classification pipeline."""

    # Fill missing numerical values with the median
    # and standardize numerical features.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Encode categories and ignore unseen categories.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Apply the appropriate preprocessing to each column.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Preprocessing followed by the supplied model.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
