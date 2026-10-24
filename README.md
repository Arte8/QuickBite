# QuickBite
food delivery case study for CAPSTONE PREP Humber 
# QuickBite — Late Delivery Prediction

## Project Overview

QuickBite is a machine learning classification project that predicts whether a food delivery order will arrive more than 10 minutes late.

The goal is to minimize the financial cost of late deliveries and customer vouchers.

## Business Costs

- Flagging an order for a voucher costs **$4**.
- Missing an actual late order costs **$15**.
- The classification threshold is optimized to minimize total intervention cost.

## Dataset

The project generates 3,000 synthetic orders using the instructor-provided `make_orders(n=3000, seed=7)` function.

The target variable is `late`, indicating whether an order is more than 10 minutes late.

The `customer_rating` column is excluded because it introduces target leakage. The `order_id` column is excluded because it is an identifier.

## Models Evaluated

Four classifiers were evaluated using five-fold stratified cross-validation:

1. DummyClassifier
2. LogisticRegression
3. BalancedLogisticRegression
4. RandomForestClassifier

The selected model is Logistic Regression, tuned using GridSearchCV.

Best hyperparameters:

- `C = 0.01`
- `solver = liblinear`

## Model Performance

The final evaluation uses a 75% training and 25% testing split.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.852 |
| Precision | 0.530 |
| Recall | 0.453 |
| F1-score | 0.488 |
| ROC-AUC | 0.796 |
| Classification threshold | 0.30 |

The selected threshold minimizes out-of-fold training intervention cost.

## Business Cost Comparison

| Policy | Test Cost |
|---|---:|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Trained model | $1,360 |

The model reduces costs by **$395 (22.5%)** compared with flagging nobody.

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/Arte8/QuickBite.git
cd QuickBite
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate quickbite-lab05
```

Install the project:

```bash
python -m pip install -e .
```

## Run the Training Pipeline

```bash
python -m latedeliveries.train
```

The training script performs cross-validation, hyperparameter tuning, threshold optimization, final evaluation, and model persistence.

## Run Automated Tests

```bash
python -m pytest -v
```

The project includes automated tests for preprocessing, prediction probabilities, missing values, unseen categories, business costs, threshold selection, and model persistence.

## Code Quality

```bash
python -m ruff format --check .
python -m ruff check .
```

## Project Outputs

- `reports/cv_results.csv` — Cross-validation results.
- `reports/metrics.json` — Final evaluation metrics and policy costs.
- `models/late_orders.joblib` — Saved trained classification pipeline.
- `models/model_info.json` — Model metadata and selected threshold.
- `notebooks/analysis.ipynb` — Visualizations and business recommendation.

## Limitations and Recommendations

The dataset is synthetic and may not fully represent real delivery operations. The model also misses some late deliveries, so its recall and financial performance should be monitored.

The model should be piloted before production deployment, with periodic evaluation of prediction quality, intervention costs, and changing delivery patterns.